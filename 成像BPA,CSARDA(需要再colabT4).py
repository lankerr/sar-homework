# -*- coding: utf-8 -*-
"""
SAR 成像算法 GPU 加速与对比版 (BPA, RDA, CSA, wKA)
- BPA: 仅提供 GPU 版本 (CUDA 加速)
- RDA/CSA/wKA: 提供 CPU/GPU 双版本对比
- 修复: wKA 相位与 ifftshift 问题
- GPU 依赖: 需要 Cupy 库
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# 检查 GPU 环境
try:
    import cupy as cp
    import cupyx.scipy.ndimage

    # 读取当前设备信息
    dev_id = cp.cuda.Device().id
    props = cp.cuda.runtime.getDeviceProperties(dev_id)
    gpu_name = props["name"].decode("utf-8")

    HAS_GPU = True
    print(f"✅ GPU 环境检测成功: {gpu_name}")
except ImportError:
    HAS_GPU = False
    print("❌ 未检测到 CuPy，无法运行 GPU 加速代码！")
    print("   请安装: pip install cupy-cuda12x  (你是 CUDA 12.8)")
    sys.exit(1)

# ==========================================
# 1. 配置参数
# ==========================================
class SARConfig:
    def __init__(self):
        # 物理常数
        self.c = 299792458.0

        # 雷达参数 (C-Band)
        self.fc = 5.3e9
        self.B = 80e6
        self.Tp = 5e-6
        self.fs = 200e6
        self.PRF = 1000.0

        # 平台几何
        self.H = 3000.0
        self.V = 150.0
        self.R0 = np.sqrt(self.H**2)
        self.L_sar = 600.0

        # 导出参数
        self.K = self.B / self.Tp
        self.lambda_c = self.c / self.fc

        # 采样窗口
        self.R_min = self.R0 - 150.0
        self.R_max = self.R0 + 150.0

        self.t_start = 2 * self.R_min / self.c
        self.t_end = 2 * self.R_max / self.c + self.Tp

        self.Nr = int((self.t_end - self.t_start) * self.fs)
        if self.Nr % 2 != 0: self.Nr += 1

        self.tr = np.linspace(self.t_start, self.t_start + self.Nr/self.fs, self.Nr, endpoint=False)

        self.Na = int(self.L_sar / self.V * self.PRF)
        self.ta = np.linspace(-self.L_sar/(2*self.V), self.L_sar/(2*self.V), self.Na)

        # 成像网格
        self.image_size = 512  # 输出分辨率
        self.scene_size = 100.0 # 场景大小 (m)

        self.x_axis = np.linspace(-self.scene_size/2, self.scene_size/2, self.image_size)
        self.y_axis = np.linspace(-self.scene_size/2, self.scene_size/2, self.image_size)

TARGETS = [[0, 0, 1.0]] # [dx, dy, rcs]

# ==========================================
# 2. 数据生成 (CPU)
# ==========================================
def simulate_data(cfg):
    print(f"📡 生成回波数据 ({cfg.Na} x {cfg.Nr})...")
    raw = np.zeros((cfg.Na, cfg.Nr), dtype=np.complex64)
    Ta, Tr = np.meshgrid(cfg.ta, cfg.tr, indexing='ij')

    for t in TARGETS:
        dx, dy, rcs = t
        pos_x = cfg.R0 + dx
        pos_y = dy

        R = np.sqrt(pos_x**2 + (pos_y - cfg.V * Ta)**2)
        tau = 2 * R / cfg.c

        mask = (np.abs(Tr - tau) <= cfg.Tp/2)
        # Baseband Phase
        phase = -4 * np.pi * cfg.fc * R / cfg.c + np.pi * cfg.K * (Tr - tau)**2
        raw += rcs * np.exp(1j * phase) * mask

    return raw

# ==========================================
# 3. 算法核心 (通用核心，支持 np/cp)
# ==========================================

# --- 工具函数 ---
def crop_center(img, cfg, xp):
    """从全尺寸结果中裁剪出成像区域"""
    if img.shape[0] == cfg.image_size and img.shape[1] == cfg.image_size:
        return img # 已经是 BPA 的物理网格

    # 理论中心索引
    r_center = int((2 * cfg.R0 / cfg.c - cfg.t_start) * cfg.fs)
    a_center = cfg.Na // 2
    w = cfg.image_size // 2

    # 简单的边界保护
    r_s = max(0, r_center - w)
    r_e = min(img.shape[0], r_center + w)
    a_s = max(0, a_center - w)
    a_e = min(img.shape[1], a_center + w)

    # img 是 (Range, Azimuth) 排列
    sub = img[r_s:r_e, a_s:a_e]

    # 放入中心
    out = xp.zeros((cfg.image_size, cfg.image_size), dtype=np.complex64)
    c_r = w - (r_center - r_s)
    c_a = w - (a_center - a_s)

    out[c_r:c_r+sub.shape[0], c_a:c_a+sub.shape[1]] = sub
    return out

# --- RDA Core ---
def rda_core(raw, cfg, xp):
    # 1. Range FFT & Compression
    fr = xp.fft.fftfreq(cfg.Nr, 1/cfg.fs)
    mf = xp.exp(1j * np.pi * fr**2 / cfg.K)
    S_rc = xp.fft.ifft(xp.fft.fft(raw, axis=1) * mf, axis=1)

    # 2. Azimuth FFT
    S_rd = xp.fft.fftshift(xp.fft.fft(S_rc, axis=0), axes=0)
    fa = xp.fft.fftshift(xp.fft.fftfreq(cfg.Na, 1/cfg.PRF))

    # 3. RCMC
    Fa, Fr = xp.meshgrid(fa, fr, indexing='ij')
    D = xp.sqrt(1 - (cfg.lambda_c * Fa / (2 * cfg.V))**2)
    dR = cfg.R0 * (1/D - 1)

    phase_rcmc = 4 * np.pi * Fr * dR / cfg.c
    phase_az = 4 * np.pi * cfg.R0 * D / cfg.lambda_c

    S_rd_rcmc = xp.fft.ifft(xp.fft.fft(S_rd, axis=1) * xp.exp(1j * phase_rcmc), axis=1)

    # 4. Azimuth Compression
    img = xp.fft.ifft(S_rd_rcmc * xp.exp(1j * phase_az), axis=0)

    return img.T # (Range, Azimuth)

# --- CSA Core ---
def csa_core(raw, cfg, xp):
    fr = xp.fft.fftfreq(cfg.Nr, 1/cfg.fs)
    fa = xp.fft.fftshift(xp.fft.fftfreq(cfg.Na, 1/cfg.PRF))

    S = xp.fft.fftshift(xp.fft.fft(xp.fft.fft(raw, axis=1), axis=0), axes=0)

    Fa, Fr = xp.meshgrid(fa, fr, indexing='ij')
    D = xp.sqrt(1 - (cfg.lambda_c * Fa / (2 * cfg.V))**2)

    # Phase Terms
    phi_mf = np.pi * Fr**2 / cfg.K
    # 避免除以0
    D_safe = D.copy()
    D_safe[D_safe < 0.1] = 0.1

    phi_src = np.pi * Fr**2 * (1/(cfg.K * D_safe) - 1/cfg.K)
    phi_rcmc = 4 * np.pi * Fr * cfg.R0 * (1/D_safe - 1) / cfg.c
    phi_ac = 4 * np.pi * cfg.R0 * D / cfg.lambda_c

    S_final = S * xp.exp(1j * (phi_mf + phi_src + phi_rcmc + phi_ac))
    img = xp.fft.ifft2(S_final)

    return img.T

# --- wKA Core (Fixed) ---
def wka_core(raw, cfg, xp):
    # 1. 2D FFT
    S2 = xp.fft.fftshift(xp.fft.fft2(raw))

    fr = xp.fft.fftshift(xp.fft.fftfreq(cfg.Nr, 1/cfg.fs))
    fa = xp.fft.fftshift(xp.fft.fftfreq(cfg.Na, 1/cfg.PRF))
    Fa, Fr = xp.meshgrid(fa, fr, indexing='ij')

    # 2. RFM
    ky_term = (cfg.c * Fa / (2 * cfg.V))**2
    kx_sq = (cfg.fc + Fr)**2 - ky_term
    kx_sq[kx_sq < 0] = 0
    kx = xp.sqrt(kx_sq)

    phi_rfm = 4 * np.pi * cfg.R0 / cfg.c * kx
    S_rfm = S2 * xp.exp(1j * phi_rfm)

    # 3. Stolt Interpolation
    ky_term_1d = (cfg.c * fa / (2*cfg.V))**2

    if xp.__name__ == 'cupy':
        # --- GPU Implementation (Vectorized Map Coordinates) ---
        # 目标: 将均匀网格 (fr) 映射回扭曲网格 (f_in)
        # 映射公式: f_in = sqrt( (f_out + fc)^2 + term ) - fc

        # 构造输出网格的坐标矩阵 (Na x Nr)
        # map_coordinates 需要的是 input array 的索引 (indices)

        # 为了利用广播，先计算 f_in 矩阵
        Fr_out = xp.broadcast_to(fr[None, :], (cfg.Na, cfg.Nr))
        Ky_term_2d = xp.broadcast_to(ky_term_1d[:, None], (cfg.Na, cfg.Nr))

        F_in = xp.sqrt((Fr_out + cfg.fc)**2 + Ky_term_2d) - cfg.fc

        # 将 F_in 转换为 Input Array 的列索引 (Range Index)
        # Fr 是均匀分布的 [-fs/2, fs/2]，对应索引 [0, Nr-1]
        # Index = (F_in - Fr_min) / df
        fr_min = fr[0]
        df = fr[1] - fr[0]
        col_indices = (F_in - fr_min) / df

        # 行索引就是 0..Na-1
        row_indices = xp.broadcast_to(xp.arange(cfg.Na)[:, None], (cfg.Na, cfg.Nr))

        # 组合坐标 [row_indices, col_indices]
        coords = xp.stack([row_indices, col_indices])

        # 实部虚部分开插值 (map_coordinates 不支持复数)
        S_stolt_real = cupyx.scipy.ndimage.map_coordinates(S_rfm.real, coords, order=1, mode='constant', cval=0)
        S_stolt_imag = cupyx.scipy.ndimage.map_coordinates(S_rfm.imag, coords, order=1, mode='constant', cval=0)
        S_stolt = S_stolt_real + 1j * S_stolt_imag

    else:
        # --- CPU Implementation (Loop) ---
        S_stolt = np.zeros_like(S_rfm)
        for i in range(cfg.Na):
            f_in = np.sqrt((fr + cfg.fc)**2 + ky_term_1d[i]) - cfg.fc
            S_stolt[i, :] = np.interp(f_in, fr, S_rfm[i, :], left=0, right=0)

    # 4. IFFT2 (Must ifftshift first!)
    S_final = xp.fft.ifftshift(S_stolt)
    img = xp.fft.ifft2(S_final)

    return img.T

# ==========================================
# 4. BPA GPU (CUDA Kernel)
# ==========================================
def bpa_gpu(raw_host, cfg):
    print("   [BPA-GPU] 启动 CUDA 内核...")

    # 准备数据
    raw_gpu = cp.array(raw_host)
    img_gpu = cp.zeros((cfg.image_size, cfg.image_size), dtype=cp.complex64)

    x_ax = cp.array(cfg.x_axis, dtype=cp.float32)
    y_ax = cp.array(cfg.y_axis, dtype=cp.float32)
    ta = cp.array(cfg.ta, dtype=cp.float32)

    # 常量传递
    c = cp.float32(cfg.c)
    fc = cp.float32(cfg.fc)
    fs = cp.float32(cfg.fs)
    R0 = cp.float32(cfg.R0)
    t_start = cp.float32(cfg.t_start)
    V = cp.float32(cfg.V)

    # CUDA Kernel
    # 每个线程计算一个像素 (i, j)
    # 循环累加所有方位向脉冲
    kernel_code = r'''
    extern "C" __global__
    void bpa_kernel(const float2* raw, float2* img,
                    const float* x_ax, const float* y_ax, const float* ta,
                    int Nx, int Ny, int Na, int Nr,
                    float c, float fc, float fs, float R0, float t_start, float V) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i >= Nx || j >= Ny) return;

        float px = x_ax[i] + R0;
        float py = y_ax[j];
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        float lambda = c / fc;
        float pi = 3.1415926535f;

        for (int k = 0; k < Na; k++) {
            float sy = V * ta[k];
            float R = sqrtf(px*px + (py - sy)*(py - sy));

            // Time index
            float delay = 2.0f * R / c;
            float idx_f = (delay - t_start) * fs;
            int idx = (int)idx_f;

            if (idx >= 0 && idx < Nr - 1) {
                // Linear Interpolation
                float frac = idx_f - (float)idx;
                float2 s0 = raw[k * Nr + idx];
                float2 s1 = raw[k * Nr + idx + 1];

                float val_re = s0.x * (1.0f - frac) + s1.x * frac;
                float val_im = s0.y * (1.0f - frac) + s1.y * frac;

                // Phase Compensation: exp(j * 4pi * R / lambda)
                // Note: Original signal has -4pi*fc*R/c. We add +4pi*fc*R/c to cancel it.
                // Or simply: phase = 4 * pi * R / lambda
                float phase = 4.0f * pi * R / lambda;
                float cp = cosf(phase);
                float sp = sinf(phase);

                // Complex Multiply: (val_re + j val_im) * (cp + j sp)
                sum_re += val_re * cp - val_im * sp;
                sum_im += val_re * sp + val_im * cp;
            }
        }

        img[i * Ny + j] = make_float2(sum_re, sum_im);
    }
    '''

    module = cp.RawModule(code=kernel_code)
    kernel = module.get_function('bpa_kernel')

    # Launch Config
    block = (16, 16)
    grid = ((cfg.image_size + 15) // 16, (cfg.image_size + 15) // 16)

    kernel(grid, block, (raw_gpu, img_gpu, x_ax, y_ax, ta,
                         cfg.image_size, cfg.image_size, cfg.Na, cfg.Nr,
                         c, fc, fs, R0, t_start, V))

    cp.cuda.Device().synchronize()
    return cp.asnumpy(img_gpu)

# ==========================================
# 5. 主程序与 Benchmark
# ==========================================
def run_benchmark():
    cfg = SARConfig()
    print("=== SAR GPU Benchmark & Comparison ===")
    print(f"Matrix: {cfg.Na} x {cfg.Nr}")
    print(f"Image : {cfg.image_size} x {cfg.image_size}")

    raw_cpu = simulate_data(cfg)
    raw_gpu = cp.array(raw_cpu)

    results = {}
    times = {'CPU': {}, 'GPU': {}}

    # --- 1. BPA (GPU Only) ---
    print("\n🔹 Running BPA (GPU)...")
    t0 = time.time()
    res_bpa = bpa_gpu(raw_cpu, cfg)
    dt = time.time() - t0
    times['GPU']['BPA'] = dt
    results['BPA'] = res_bpa
    print(f"   Time: {dt:.4f}s")

    # --- 2. Generic Algorithms (CPU vs GPU) ---
    algos = [
        ('RDA', rda_core),
        ('CSA', csa_core),
        ('wKA', wka_core)
    ]

    for name, func in algos:
        print(f"\n🔹 Running {name}...")

        # CPU Run
        print("   [CPU] Running...", end='')
        t0 = time.time()
        res_cpu_full = func(raw_cpu, cfg, np)
        res_cpu = crop_center(res_cpu_full, cfg, np)
        dt_cpu = time.time() - t0
        times['CPU'][name] = dt_cpu
        print(f" Done ({dt_cpu:.4f}s)")

        # GPU Run
        print("   [GPU] Running...", end='')
        cp.cuda.Device().synchronize()
        t0 = time.time()
        res_gpu_full = func(raw_gpu, cfg, cp)
        res_gpu = crop_center(res_gpu_full, cfg, cp)
        cp.cuda.Device().synchronize()
        dt_gpu = time.time() - t0
        times['GPU'][name] = dt_gpu
        results[name] = cp.asnumpy(res_gpu) # Save GPU result for plotting
        print(f" Done ({dt_gpu:.4f}s)")
        print(f"   🚀 Speedup: {dt_cpu/dt_gpu:.1f}x")

    # --- Visualization ---
    print("\n📊 Plotting Results...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Images (Best versions)
    img_list = [results['BPA'], results['RDA'], results['CSA'], results['wKA']]
    names = ['BPA (GPU)', 'RDA (GPU)', 'CSA (GPU)', 'wKA (GPU)']

    for i, (img, title) in enumerate(zip(img_list, names)):
        ax = axes.flat[i]
        amp = np.abs(img)
        mx = np.max(amp)
        # Log Scale
        if mx > 0:
            amp_log = 20 * np.log10(amp / mx + 1e-9)
        else:
            amp_log = np.zeros_like(amp)

        im = ax.imshow(amp_log, cmap='jet', vmin=-30, vmax=0, origin='lower',
                       extent=[cfg.y_axis[0], cfg.y_axis[-1], cfg.x_axis[0], cfg.x_axis[-1]])
        ax.set_title(title + f"\nMax: {mx:.1f}")
        plt.colorbar(im, ax=ax, fraction=0.046)
        # Mark center
        ax.plot(0, 0, 'w+', ms=15)

    # Row 2: Performance Chart
    ax_perf = axes[1, 1]
    # Hide unused subplots
    axes[1, 0].axis('off')
    axes[1, 2].axis('off')

    labels = ['RDA', 'CSA', 'wKA']
    cpu_times = [times['CPU'][l] for l in labels]
    gpu_times = [times['GPU'][l] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    ax_perf.bar(x - width/2, cpu_times, width, label='CPU')
    ax_perf.bar(x + width/2, gpu_times, width, label='GPU')

    # Add BPA GPU bar separately
    ax_perf.bar(len(labels), times['GPU']['BPA'], width, label='BPA (GPU)', color='green')

    ax_perf.set_ylabel('Time (s)')
    ax_perf.set_title('Execution Time Comparison')
    ax_perf.set_xticks(list(x) + [len(labels)])
    ax_perf.set_xticklabels(labels + ['BPA'])
    ax_perf.legend()
    ax_perf.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_benchmark()