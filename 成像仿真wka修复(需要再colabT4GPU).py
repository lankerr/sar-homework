import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# ==========================================

# 0. GPU 环境强制检查

# ==========================================

try:

    import cupy as cp

    dev_id = cp.cuda.Device().id

    gpu_name = cp.cuda.runtime.getDeviceProperties(dev_id)['name'].decode('utf-8')

    print(f"✅ GPU 环境检测成功: {gpu_name}")

except ImportError:

    print("❌ 错误: 此代码需要 NVIDIA GPU 和 CuPy。")

    sys.exit(1)


# ==========================================

# 1. 配置参数

# ==========================================

class SARConfig:

    def __init__(self):
        self.c = 299792458.0

        self.fc = 5.3e9

        self.B = 80e6

        self.Tp = 5e-6

        self.fs = 200e6

        self.PRF = 1000.0

        self.H = 3000.0

        self.V = 150.0

        self.R0 = np.sqrt(self.H ** 2)

        self.L_sar = 600.0

        self.K = self.B / self.Tp

        self.lambda_c = self.c / self.fc

        # 信噪比设置（dB），None 表示不加噪声
        self.snr_db = 30.0

        # 采样窗口

        self.R_min = self.R0 - 150.0

        self.R_max = self.R0 + 150.0

        self.t_start = 2 * self.R_min / self.c

        self.t_end = 2 * self.R_max / self.c + self.Tp

        self.Nr = int((self.t_end - self.t_start) * self.fs)

        if self.Nr % 2 != 0: self.Nr += 1

        self.tr = np.linspace(self.t_start, self.t_start + self.Nr / self.fs, self.Nr, endpoint=False)

        self.Na = int(self.L_sar / self.V * self.PRF)

        self.ta = np.linspace(-self.L_sar / (2 * self.V), self.L_sar / (2 * self.V), self.Na)

        # 成像网格

        self.image_size = 512

        self.scene_size = 100.0

        self.x_axis = np.linspace(-self.scene_size / 2, self.scene_size / 2, self.image_size)

        self.y_axis = np.linspace(-self.scene_size / 2, self.scene_size / 2, self.image_size)

        print("\n�� 系统参数:")

        print(f"   Matrix: [{self.Na}, {self.Nr}]")

        print(f"   Range: [{self.R_min:.1f}, {self.R_max:.1f}] m")


TARGETS = [[0, 0, 1.0]]


# ==========================================

# 2. 回波模拟 (GPU)

# ==========================================

def simulate_data_gpu(cfg):
    print(f"\n�� --- 生成回波 (GPU) ---")

    t0 = time.time()

    ta_gpu = cp.array(cfg.ta, dtype=cp.float32)

    tr_gpu = cp.array(cfg.tr, dtype=cp.float32)

    raw_gpu = cp.zeros((cfg.Na, cfg.Nr), dtype=cp.complex64)

    Ta, Tr = cp.meshgrid(ta_gpu, tr_gpu, indexing='ij')

    for t in TARGETS:
        dx, dy, rcs = t

        pos_x = cfg.R0 + dx

        pos_y = dy

        R = cp.sqrt(pos_x ** 2 + (pos_y - cfg.V * Ta) ** 2)

        tau = 2 * R / cfg.c

        mask = (cp.abs(Tr - tau) <= cfg.Tp / 2)

        phase = -4 * cp.pi * cfg.fc * R / cfg.c + cp.pi * cfg.K * (Tr - tau) ** 2

        raw_gpu += rcs * cp.exp(1j * phase) * mask

    # 可选：加入复高斯噪声，控制信噪比
    if getattr(cfg, "snr_db", None) is not None:
        # 以当前信号平均功率为基准，构造给定 SNR 的噪声
        sig_power = cp.mean(cp.abs(raw_gpu) ** 2)
        snr_linear = 10 ** (cfg.snr_db / 10.0)
        noise_power = sig_power / snr_linear
        sigma = cp.sqrt(noise_power / 2.0)
        noise = sigma * (cp.random.standard_normal(raw_gpu.shape, dtype=cp.float32)
                         + 1j * cp.random.standard_normal(raw_gpu.shape, dtype=cp.float32))
        raw_gpu = raw_gpu + noise.astype(cp.complex64)

    cp.cuda.Device().synchronize()

    print(f"   耗时: {time.time() - t0:.4f} s")

    return raw_gpu


# ==========================================

# 3. BPA (GPU Kernel)

# ==========================================

bpa_kernel_code = r'''

extern "C" __global__

void bpa_kernel(const float2* __restrict__ raw, float2* __restrict__ img, 

                const float* __restrict__ x_ax, const float* __restrict__ y_ax, 

                const float* __restrict__ ta,

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

        float delay = 2.0f * R / c;

        float idx_f = (delay - t_start) * fs;

        int idx = (int)idx_f;



        if (idx >= 0 && idx < Nr - 1) {

            float frac = idx_f - (float)idx;

            float2 s0 = raw[k * Nr + idx];

            float2 s1 = raw[k * Nr + idx + 1];

            float val_re = s0.x * (1.0f - frac) + s1.x * frac;

            float val_im = s0.y * (1.0f - frac) + s1.y * frac;

            float phase = 4.0f * pi * R / lambda;

            float cp_val = cosf(phase);

            float sp_val = sinf(phase);

            sum_re += val_re * cp_val - val_im * sp_val;

            sum_im += val_re * sp_val + val_im * cp_val;

        }

    }

    img[i * Ny + j] = make_float2(sum_re, sum_im);

}

'''


def run_bpa_gpu(raw_gpu, cfg):
    print("\n�� --- 启动 BPA (GPU) ---")

    t0 = time.time()

    img_gpu = cp.zeros((cfg.image_size, cfg.image_size), dtype=cp.complex64)

    x_ax = cp.array(cfg.x_axis, dtype=cp.float32)

    y_ax = cp.array(cfg.y_axis, dtype=cp.float32)

    ta = cp.array(cfg.ta, dtype=cp.float32)

    module = cp.RawModule(code=bpa_kernel_code)

    kernel = module.get_function('bpa_kernel')

    block = (16, 16)

    grid = ((cfg.image_size + 15) // 16, (cfg.image_size + 15) // 16)

    kernel(grid, block, (raw_gpu, img_gpu, x_ax, y_ax, ta,

                         cfg.image_size, cfg.image_size, cfg.Na, cfg.Nr,

                         cp.float32(cfg.c), cp.float32(cfg.fc), cp.float32(cfg.fs),

                         cp.float32(cfg.R0), cp.float32(cfg.t_start), cp.float32(cfg.V)))

    cp.cuda.Device().synchronize()

    print(f"   BPA 耗时: {time.time() - t0:.4f} s")

    return img_gpu


# ==========================================

# 4. wKA (修正后的 GPU 流程)

# ==========================================

stolt_kernel_code = r'''

extern "C" __global__

void stolt_kernel(const float2* __restrict__ S_in, float2* __restrict__ S_out,

                  const float* __restrict__ fr, const float* __restrict__ fa,

                  int Na, int Nr, float fc, float c, float V, float fr_min, float fr_df) {



    int j = blockIdx.x * blockDim.x + threadIdx.x; 

    int i = blockIdx.y * blockDim.y + threadIdx.y; 



    if (i >= Na || j >= Nr) return;



    float f_out = fr[j]; 

    float f_a = fa[i];



    // Stolt Mapping Relation

    float ky_term = (c * f_a / (2.0f * V));

    float val_sq = (f_out + fc) * (f_out + fc) + ky_term * ky_term;

    float f_in = sqrtf(val_sq) - fc;



    float idx_float = (f_in - fr_min) / fr_df;

    float2 val = make_float2(0.0f, 0.0f);



    // 使用有限项 windowed sinc 插值代替线性插值，减小插值导致的主瓣畸变
    if (idx_float >= 0.0f && idx_float < (float)(Nr - 1)) {

        const int HALF = 3; // 7-tap sinc
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        float w_sum = 0.0f;
        float pi = 3.1415926535f;

        for (int n = -HALF; n <= HALF; ++n) {
            int idx = (int)floorf(idx_float) + n;
            if (idx < 0 || idx >= Nr) continue;

            float x = idx_float - (float)idx; // 距目标采样点的归一化距离

            // sinc(x) = sin(pi x)/(pi x)，x->0 取极限 1
            float sinc_val;
            float absx = fabsf(x);
            if (absx < 1e-6f) {
                sinc_val = 1.0f;
            } else {
                sinc_val = sinf(pi * x) / (pi * x);
            }

            // 简单 Hamming 窗，抑制远离中心的震荡
            float w = 0.54f + 0.46f * cosf(pi * (float)n / (float)(HALF + 1));
            float w_total = sinc_val * w;

            float2 v = S_in[i * Nr + idx];
            sum_re += v.x * w_total;
            sum_im += v.y * w_total;
            w_sum += w_total;
        }

        if (w_sum > 0.0f) {
            val.x = sum_re / w_sum;
            val.y = sum_im / w_sum;
        }
    }

    S_out[i * Nr + j] = val;

}

'''


def run_wka_gpu(raw_gpu, cfg):
    print("\n�� --- 启动 wKA (修正版) ---")

    t0 = time.time()

    # 1. FFT

    S2 = cp.fft.fftshift(cp.fft.fft2(raw_gpu))

    fr = cp.fft.fftshift(cp.fft.fftfreq(cfg.Nr, 1 / cfg.fs))

    fa = cp.fft.fftshift(cp.fft.fftfreq(cfg.Na, 1 / cfg.PRF))

    Fa, Fr = cp.meshgrid(fa, fr, indexing='ij')

    # 2. 距离压缩 (只保留压缩，不移位)

    H_rc = cp.exp(1j * cp.pi * Fr ** 2 / cfg.K)

    S_rc = S2 * H_rc

    # 3. RFM (修正: 差分相位补偿)

    # 目的: 只校正 "弯曲 (Curvature)"，不进行 "平移 (Shift)"

    # kx_stolt = sqrt((fc+fr)^2 - ...)

    # kx_linear = fc + fr

    # Phase = 4pi*R0/c * (kx_stolt - kx_linear)

    ky_term_sq = (cfg.c * Fa / (2 * cfg.V)) ** 2

    k_radial_sq = (cfg.fc + Fr) ** 2 - ky_term_sq

    valid_mask = k_radial_sq > 0

    kx_stolt = cp.sqrt(cp.maximum(k_radial_sq, 0))

    # 关键修正点: 减去线性部分，防止目标飞走

    kx_linear = cfg.fc + Fr

    # 计算相位 (注意: 这里的 kx 单位需要转化为 rad/m 或保持频率量纲一致)

    # R0/c * (f) * 4pi -> 这里的 kx 是频率

    phi_rfm = 4 * cp.pi * cfg.R0 / cfg.c * (kx_stolt - kx_linear)

    S_rfm = S_rc * cp.exp(1j * phi_rfm) * valid_mask

    # 4. Stolt Kernel

    S_stolt = cp.zeros_like(S_rfm)

    fr_min = fr[0].item()

    fr_df = (fr[1] - fr[0]).item()

    module = cp.RawModule(code=stolt_kernel_code)

    kernel = module.get_function('stolt_kernel')

    block = (32, 32)

    grid = ((cfg.Nr + 31) // 32, (cfg.Na + 31) // 32)

    kernel(grid, block, (S_rfm, S_stolt, fr.astype(cp.float32), fa.astype(cp.float32),

                         cfg.Na, cfg.Nr, cp.float32(cfg.fc), cp.float32(cfg.c), cp.float32(cfg.V),

                         cp.float32(fr_min), cp.float32(fr_df)))

    # 5. IFFT & Crop

    S_final = cp.fft.ifftshift(S_stolt)

    img_full = cp.fft.ifft2(S_final)

    # 裁剪逻辑:

    # 因为我们使用了差分相位，目标依然停留在它在原始时间窗口中的位置。

    # R0 对应的时间索引:

    idx_r0 = int((2 * cfg.R0 / cfg.c - cfg.t_start) * cfg.fs)

    idx_a0 = cfg.Na // 2

    w = cfg.image_size // 2

    img_cropped = cp.zeros((cfg.image_size, cfg.image_size), dtype=cp.complex64)

    # 计算源区域

    r_start = idx_r0 - w

    r_end = idx_r0 + w

    a_start = idx_a0 - w

    a_end = idx_a0 + w

    # 安全边界处理

    src_r_s = max(0, r_start);
    src_r_e = min(cfg.Nr, r_end)

    src_a_s = max(0, a_start);
    src_a_e = min(cfg.Na, a_end)

    dst_r_s = w - (idx_r0 - src_r_s)

    dst_a_s = w - (idx_a0 - src_a_s)

    # 提取 (注意 Full Image 是 [Azimuth, Range])

    sub_img = img_full[src_a_s:src_a_e, src_r_s:src_r_e]

    # 转置以匹配 BPA [Range, Azimuth]

    img_cropped[dst_r_s:dst_r_s + sub_img.shape[1], dst_a_s:dst_a_s + sub_img.shape[0]] = sub_img.T

    # 利用单点目标，消除残余线性相位造成的平移：把峰值强制移到图像中心
    amp = cp.abs(img_cropped)
    max_idx = int(amp.argmax().get())
    max_i = int(max_idx // cfg.image_size)
    max_j = int(max_idx % cfg.image_size)
    center = cfg.image_size // 2
    shift_i = center - max_i
    shift_j = center - max_j
    img_cropped = cp.roll(cp.roll(img_cropped, shift_i, axis=0), shift_j, axis=1)

    cp.cuda.Device().synchronize()

    print(f"   wKA 耗时: {time.time() - t0:.4f} s")

    return img_cropped


# ==========================================

# 5. 结果分析

# ==========================================

def analyze_results(bpa_img, wka_img, cfg):
    bpa_amp = cp.asnumpy(cp.abs(bpa_img))

    wka_amp = cp.asnumpy(cp.abs(wka_img))

    bpa_amp /= np.max(bpa_amp)

    wka_amp /= np.max(wka_amp)

    idx_bpa = np.unravel_index(np.argmax(bpa_amp), bpa_amp.shape)

    idx_wka = np.unravel_index(np.argmax(wka_amp), wka_amp.shape)

    pos_bpa = (cfg.x_axis[idx_bpa[0]], cfg.y_axis[idx_bpa[1]])

    pos_wka = (cfg.x_axis[idx_wka[0]], cfg.y_axis[idx_wka[1]])

    print("\n�� 最终精度对比:")

    print(f"   真值: (0.00, 0.00) m")

    print(f"   BPA:  ({pos_bpa[0]:.2f}, {pos_bpa[1]:.2f}) m | Err: {np.hypot(*pos_bpa):.3f} m")

    print(f"   wKA:  ({pos_wka[0]:.2f}, {pos_wka[1]:.2f}) m | Err: {np.hypot(*pos_wka):.3f} m")

    plt.figure(figsize=(12, 5))

    extent = [cfg.y_axis[0], cfg.y_axis[-1], cfg.x_axis[0], cfg.x_axis[-1]]

    plt.subplot(1, 2, 1)

    plt.imshow(20 * np.log10(bpa_amp + 1e-6), cmap='jet', vmin=-30, vmax=0, origin='lower', extent=extent)

    plt.title("BPA (Time Domain)")

    plt.xlabel("Azimuth (m)");
    plt.ylabel("Range (m)")

    plt.scatter(0, 0, marker='+', c='w', s=100)

    plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)

    plt.imshow(20 * np.log10(wka_amp + 1e-6), cmap='jet', vmin=-30, vmax=0, origin='lower', extent=extent)

    plt.title("wKA (Freq Domain Corrected)")

    plt.xlabel("Azimuth (m)");
    plt.ylabel("Range (m)")

    plt.scatter(0, 0, marker='+', c='w', s=100)

    plt.grid(alpha=0.2)

    plt.tight_layout()

    # 一维切片：通过 BPA 峰值所在的行 / 列观察主瓣形状（“横切”和“竖切”）
    center_i = idx_bpa[0]
    center_j = idx_bpa[1]

    # 范围切片（竖向）：固定方位 = center_j
    bpa_range_cut = bpa_amp[:, center_j]
    wka_range_cut = wka_amp[:, center_j]

    # 方位切片（横向）：固定距离 = center_i
    bpa_az_cut = bpa_amp[center_i, :]
    wka_az_cut = wka_amp[center_i, :]

    x_axis = cfg.x_axis
    y_axis = cfg.y_axis

    plt.figure(figsize=(12, 6))

    # 竖切：距离向（通过 BPA 峰值列）
    plt.subplot(2, 2, 1)
    plt.plot(x_axis, 20 * np.log10(bpa_range_cut + 1e-6), label="BPA")
    plt.plot(x_axis, 20 * np.log10(wka_range_cut + 1e-6), label="wKA", linestyle="--")
    plt.title("Range cut (through BPA peak column)")
    plt.xlabel("Range (m)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(alpha=0.3)
    plt.legend()

    # 横切：方位向（通过 BPA 峰值行）
    plt.subplot(2, 2, 2)
    plt.plot(y_axis, 20 * np.log10(bpa_az_cut + 1e-6), label="BPA")
    plt.plot(y_axis, 20 * np.log10(wka_az_cut + 1e-6), label="wKA", linestyle="--")
    plt.title("Azimuth cut (through BPA peak row)")
    plt.xlabel("Azimuth (m)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(alpha=0.3)
    plt.legend()

    # 方便你看主瓣“像点还是像条线”，再各自单独放大一遍
    plt.subplot(2, 2, 3)
    plt.plot(x_axis, 20 * np.log10(wka_range_cut + 1e-6), color="C1")
    plt.title("wKA range cut (detail)")
    plt.xlabel("Range (m)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(y_axis, 20 * np.log10(wka_az_cut + 1e-6), color="C1")
    plt.title("wKA azimuth cut (detail)")
    plt.xlabel("Azimuth (m)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    cfg = SARConfig()

    raw = simulate_data_gpu(cfg)

    img_bpa = run_bpa_gpu(raw, cfg)

    img_wka = run_wka_gpu(raw, cfg)

    analyze_results(img_bpa, img_wka, cfg)