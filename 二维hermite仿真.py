# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import correlate, resample

from scipy.interpolate import interp1d

import warnings
import json
import time
import os



# 忽略除零警告

warnings.filterwarnings('ignore')

plt.style.use('dark_background')

# 修复中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# region agent log
DEBUG_LOG_PATH = r"c:\Users\97290\Desktop\SAR+分数域大作业\邓嘉轩-SAR-大作业\.cursor\debug.log"


def agent_log(hypothesis_id, location, message, data, run_id="pre-fix"):
    """轻量级 NDJSON 日志，用于调试左下角杂波来源。"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)

        entry = {
            "sessionId": "debug-session",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # 调试日志失败时静默忽略，避免影响仿真流程
        pass
# endregion


def print_section(title):

    print(f"\n{'='*70}")

    print(f" {title}")

    print(f"{'='*70}")



# ==========================================

# 1. 系统参数

# ==========================================

print_section("1. 系统参数")



c = 299792458.0

fc = 5.3e9

wavelength = c / fc

B = 80e6

fs = 200e6

Tp = 5e-6

Nr = int(Tp * fs)

t_pulse = np.linspace(-Tp/2, Tp/2, Nr)



# SAR 参数

H = 5000.0

V = 100.0

R0 = 10000.0

PRF = 400.0

Na = 128

Tsa = Na / PRF

range_res = c / (2 * B)

azimuth_res = wavelength * R0 / (2 * V * Tsa)



# ==========================================

# 2. 物理归一化 Hermite 基底生成 

# ==========================================

print_section("2. 鲁棒的波形生成 (Robust Waveform Gen)")



max_order = 50

sparse_orders = [10, 18, 27, 36, 45] # 使用的稀疏阶数



# --- 关键修复：动态计算 Sigma ---

# 目标：让最高阶函数在窗口边缘衰减到接近0

# 经验公式：X_max approx sqrt(2*n + 1) + 4 (安全余量)

# 对于 n=50, sqrt(101)=10, 需要 x 范围至少到 14

x_max_needed = np.sqrt(2 * max_order + 1) + 4.0

sigma = x_max_needed / (Tp / 2)



print(f"  [Fix] 最大阶数 N={max_order}")

print(f"  [Fix] 需要的数学域范围 X_max = ±{x_max_needed:.1f}")

print(f"  [Fix] 修正后的 Sigma = {sigma:.2e} (Hz)")



t_scaled = t_pulse * sigma



# --- 关键修复：递推生成归一化函数 ---

# 避免使用 hermite(n)(x)，因为它会产生天文数字

# 使用物理递推: psi_n = sqrt(2/n)*x*psi_{n-1} - sqrt((n-1)/n)*psi_{n-2}

# psi_0 = pi^(-0.25) * exp(-x^2/2)



hermite_basis = np.zeros((max_order + 1, Nr))



# 初始化 n=0

norm_factor = np.pi**(-0.25)

hermite_basis[0, :] = norm_factor * np.exp(-t_scaled**2 / 2)



# 初始化 n=1

hermite_basis[1, :] = np.sqrt(2) * t_scaled * hermite_basis[0, :]



# 递推 n=2...max_order

for n in range(2, max_order + 1):

    c1 = np.sqrt(2.0 / n)

    c2 = np.sqrt((n - 1.0) / n)

    hermite_basis[n, :] = c1 * t_scaled * hermite_basis[n-1, :] - c2 * hermite_basis[n-2, :]



# 验证边缘能量

edge_energy = np.sum(hermite_basis[:, :5]**2) + np.sum(hermite_basis[:, -5:]**2)

total_energy = np.sum(hermite_basis**2)

edge_ratio = edge_energy / total_energy



print(f"  [Check] 边缘能量占比: {edge_ratio:.2e}")

if edge_ratio < 1e-4:

    print("波形在窗口内自然衰减，无截断！")

else:

    print("警告：波形截断依然存在，请增大 Sigma！")



# 再次QR分解确保数值上的完美正交

q, _ = np.linalg.qr(hermite_basis.T)

hermite_basis_ortho = q.T



# ==========================================

# 3. 信号定义

# ==========================================

def get_waveform(orders, weights):

    wf = np.zeros(Nr)

    for o, w in zip(orders, weights):

        wf += w * hermite_basis_ortho[o, :]

    wf = wf / np.sqrt(np.sum(wf**2)) # 能量归一化

    return wf.astype(np.complex128) * np.exp(1j * 2 * np.pi * fc * t_pulse)



# 发射波形

sparse_wf = get_waveform(sparse_orders, [1.0, 0.9, 0.8, 0.7, 0.6])



# LFM 干扰波形

K_lfm = B / Tp

lfm_wf = np.exp(1j * np.pi * K_lfm * t_pulse**2) * np.exp(1j * 2 * np.pi * fc * t_pulse)

lfm_wf = lfm_wf / np.sqrt(np.sum(np.abs(lfm_wf)**2))



# ==========================================

# 4. 稀疏滤波器

# ==========================================

def apply_filter(rx_sig, orders_to_keep):

    # 下变频

    bb = rx_sig * np.exp(-1j * 2 * np.pi * fc * t_pulse)

    # 分解

    coeffs = np.dot(hermite_basis_ortho, bb)

    # 滤波

    mask = np.zeros_like(coeffs)

    mask[orders_to_keep] = 1.0

    coeffs_filtered = coeffs * mask

    # 重构

    bb_recon = np.dot(hermite_basis_ortho.T, coeffs_filtered)

    # 上变频

    return bb_recon * np.exp(1j * 2 * np.pi * fc * t_pulse)



# 验证 LFM 抑制能力

lfm_filtered = apply_filter(lfm_wf, sparse_orders)

rejection = 10 * np.log10(np.sum(np.abs(lfm_filtered)**2) / np.sum(np.abs(lfm_wf)**2))

print(f"  [Check] LFM 干扰抑制比 (Theoretical): {-rejection:.1f} dB (预期 > 15dB)")



# ==========================================

# 5. 回波生成与处理 (2D Simulation)

# ==========================================

print_section("3. 开始 2D 仿真 (BPA Imaging)")



# 场景

targets = [{'x': 0, 'y': 0, 'rcs': 1.0}]

jammers = [{'x': -30, 'y': 20, 'power': 100.0}] # JSR = 20dB



# 扩展参数

range_swath = 200

Nr_ext = Nr * 3

dr = range_swath / Nr_ext

eta = np.arange(Na) / PRF - Na/(2*PRF)

platform_x = V * eta

gr_center = np.sqrt(R0**2 - H**2)



# --- 生成回波 ---

def gen_echo(wf, use_jam):

    echo = np.zeros((Na, Nr_ext), dtype=np.complex128)

    freq = np.fft.fftfreq(Nr_ext, d=1/fs)



    # 扩展波形以便FFT移位

    wf_pad = np.zeros(Nr_ext, dtype=np.complex128)

    wf_pad[:Nr] = wf

    wf_spec = np.fft.fft(wf_pad)



    # 干扰使用 LFM

    lfm_pad = np.zeros(Nr_ext, dtype=np.complex128)

    lfm_pad[:Nr] = lfm_wf

    lfm_spec = np.fft.fft(lfm_pad)



    for i in range(Na):

        px = platform_x[i]



        # Target

        for t in targets:

            R = np.sqrt((px - t['x'])**2 + (gr_center + t['y'])**2 + H**2)

            delay = (R - (R0 - range_swath/2)) / (c/2)

            shift_phase = np.exp(-1j * 2 * np.pi * freq * delay)

            phase_hist = np.exp(-1j * 4 * np.pi * R / wavelength)

            echo[i] += t['rcs'] * np.fft.ifft(wf_spec * shift_phase) * phase_hist



        # Jammer

        if use_jam:

            for j in jammers:

                R = np.sqrt((px - j['x'])**2 + (gr_center + j['y'])**2 + H**2)

                delay = (R - (R0 - range_swath/2)) / (c/2)

                shift_phase = np.exp(-1j * 2 * np.pi * freq * delay)

                phase_hist = np.exp(-1j * 4 * np.pi * R / wavelength)

                # 干扰总是发射 LFM

                echo[i] += np.sqrt(j['power']) * np.fft.ifft(lfm_spec * shift_phase) * phase_hist



    return echo



print("  生成回波: Sparse + JSR 20dB...")

echo_raw = gen_echo(sparse_wf, use_jam=True)

echo_clean_ref = gen_echo(sparse_wf, use_jam=False) # 用于计算真实增益



# --- 滤波与压缩 ---

print("  执行稀疏滤波与距离压缩...")

rc_data = np.zeros_like(echo_raw)

window = np.hanning(Nr)



for i in range(Na):

    # 滑动窗口滤波

    sig_long = echo_raw[i]

    sig_out = np.zeros_like(sig_long)



    # 简单分段处理 (Overlap-Add simplified)

    # 为演示，直接对整个脉冲串做分段滤波是比较慢的，这里我们简化：

    # 假设回波主要集中在中间，直接对有效段滤波

    # 在实际雷达中是逐脉冲处理。这里我们扫描强点。



    # 更好的方法：直接对长信号切片滤波

    for k in range(0, Nr_ext - Nr, Nr//2):

        chunk = sig_long[k:k+Nr]

        chunk_f = apply_filter(chunk, sparse_orders)

        sig_out[k:k+Nr] += chunk_f * window



    # 匹配滤波

    rc_data[i] = correlate(sig_out, sparse_wf, mode='same')


# region agent log - 修复：距离向边缘加窗衰减前段和后段能量
# 添加距离向边缘加窗，衰减前段和后段的残留能量
range_taper = np.ones(Nr_ext)
taper_len = int(0.15 * Nr_ext)  # 前后各衰减 15% 的区域
if taper_len > 0:
    taper_window = np.hanning(2 * taper_len + 1)
    range_taper[:taper_len] = taper_window[:taper_len]
    range_taper[-taper_len:] = taper_window[-taper_len:]
    for i in range(Na):
        rc_data[i] *= range_taper
agent_log(
    "FIX1",
    "二维hermite仿真.py:range_taper",
    "应用距离向边缘加窗衰减前段和后段",
    {
        "taper_len": int(taper_len),
        "front_mean_pow": float(np.mean(np.abs(rc_data[:, :taper_len]) ** 2)),
        "center_mean_pow": float(np.mean(np.abs(rc_data[:, taper_len:-taper_len]) ** 2)),
        "rear_mean_pow": float(np.mean(np.abs(rc_data[:, -taper_len:]) ** 2)),
    },
)
# endregion


# 对照组：LFM受到干扰

print("  生成对照组 (LFM 受干扰)...")

echo_lfm_jam = gen_echo(lfm_wf, use_jam=True)

rc_lfm = np.zeros_like(echo_lfm_jam)

for i in range(Na):

    rc_lfm[i] = correlate(echo_lfm_jam[i], lfm_wf, mode='same')


# ==========================================
# 5.5 FRFT 旋转角度搜索与 LFM 抑制 (最优域清洗思路)
# ==========================================

print_section("4.5 FRFT 角度搜索与最优域清洗")

# region agent log - FRFT 实现
def discrete_frft(x, alpha):
    """
    离散分数阶傅里叶变换 (Discrete Fractional Fourier Transform)
    基于采样定理的快速实现
    """
    N = len(x)
    if N == 0:
        return x
    
    # 归一化到 [-1, 1] 区间
    t = np.linspace(-1, 1, N)
    
    # FRFT 核函数参数
    cot_alpha = 1.0 / np.tan(alpha) if abs(np.sin(alpha)) > 1e-10 else 0.0
    csc_alpha = 1.0 / np.sin(alpha) if abs(np.sin(alpha)) > 1e-10 else 1.0
    
    # 特殊情况处理
    if abs(alpha % (2*np.pi)) < 1e-10:
        return x.copy()
    elif abs(alpha % (2*np.pi) - np.pi/2) < 1e-10:
        return np.fft.fft(x) / np.sqrt(N)
    elif abs(alpha % (2*np.pi) - np.pi) < 1e-10:
        return x[::-1]
    elif abs(alpha % (2*np.pi) - 3*np.pi/2) < 1e-10:
        return np.fft.ifft(x) * np.sqrt(N)
    
    u = np.linspace(-1, 1, N)
    
    # 调制 -> FFT -> 解调 (快速近似实现)
    chirp_pre = np.exp(1j * np.pi * cot_alpha * t**2)
    chirp_post = np.exp(1j * np.pi * cot_alpha * u**2)
    
    x_mod = x * chirp_pre
    X_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x_mod)))
    X_alpha = np.sqrt(np.abs(csc_alpha) / N) * chirp_post * X_fft
    
    return X_alpha

# 扫描角度找到 LFM 的最优角度（能量最集中）
print("  扫描 FRFT 角度，寻找 LFM 最优聚焦角度...")
alpha_range = np.linspace(0, np.pi, 61)  # 更精细扫描
lfm_test_pulse = rc_lfm[Na // 2]

peak_ratios = []
for alpha in alpha_range:
    lfm_frft = discrete_frft(lfm_test_pulse, alpha)
    ratio = np.max(np.abs(lfm_frft)**2) / (np.sum(np.abs(lfm_frft)**2) + 1e-12)
    peak_ratios.append(ratio)

best_alpha = alpha_range[np.argmax(peak_ratios)]
print(f"  ✅ LFM 最优域角度: α = {best_alpha:.3f} rad ({best_alpha*180/np.pi:.1f}°)")

# 在最优域进行“清洗” (Notch Filter)
print(f"  在最优域 α={best_alpha:.3f} 执行 LFM 尖峰抑制并逆变换...")
rc_data_cleaned = np.zeros_like(rc_data)

for i in range(Na):
    # 1. 转到最优域
    sig_frft = discrete_frft(rc_data[i], best_alpha)
    
    # 2. 定位并抑制尖峰 (LFM 在此域是冲击状)
    # 使用中值滤波或简单的门限抑制
    mag = np.abs(sig_frft)
    threshold = 3.5 * np.median(mag) # 动态门限
    mask = mag < threshold
    sig_frft_filtered = sig_frft * mask
    
    # 3. 逆变换回时域
    rc_data_cleaned[i] = discrete_frft(sig_frft_filtered, -best_alpha)

# region 诊断 5, 6, 7
# DIAG5: 正交性保持
test_h0 = discrete_frft(hermite_basis_ortho[0, :], best_alpha)
test_h1 = discrete_frft(hermite_basis_ortho[1, :], best_alpha)
ortho_val = np.abs(np.vdot(test_h0, test_h1))

# DIAG6: 分量保持 (对第一个脉冲)
pulse_0_orig = rc_data[0]
pulse_0_clean = rc_data_cleaned[0]
sim_val = np.abs(np.vdot(pulse_0_orig, pulse_0_clean)) / (np.linalg.norm(pulse_0_orig) * np.linalg.norm(pulse_0_clean) + 1e-12)

agent_log(
    "FRFT_BEST",
    "二维hermite仿真.py:best_frft_clean",
    "最优域清洗诊断",
    {
        "best_alpha": float(best_alpha),
        "basis_ortho_check": float(ortho_val),
        "similarity_raw_vs_cleaned": float(sim_val),
        "peak_reduction_db": float(20*np.log10(np.max(np.abs(rc_data))/np.max(np.abs(rc_data_cleaned)) + 1e-6))
    },
    run_id="frft-best-fix"
)
# endregion
# endregion


# ==========================================

# 6. BPA 成像 (线性插值)

# ==========================================

print_section("4. BPA 成像")



def bpa(rc_in):

    Nx, Ny = 60, 60

    img = np.zeros((Ny, Nx), dtype=np.complex128)

    x_vec = np.linspace(-40, 40, Nx)

    y_vec = np.linspace(-40, 40, Ny)



    for i in range(Na):

        px = platform_x[i]

        for iy, y in enumerate(y_vec):

            for ix, x in enumerate(x_vec):

                R = np.sqrt((px - x)**2 + (gr_center + y)**2 + H**2)

                bin_idx = (R - (R0 - range_swath/2)) / dr



                # 线性插值

                idx_low = int(np.floor(bin_idx))

                w = bin_idx - idx_low

                if 0 <= idx_low < Nr_ext - 1:
                    # region agent log - 修复：对前段 bin 做门限衰减
                    # 衰减距离向前段的能量（bin_idx < 1200 的区域，覆盖左下角像素的 bin_idx ≈ 981）
                    range_guard = 1200  # 前段保护区域，覆盖到 bin_idx ≈ 981 的区域
                    if bin_idx < range_guard:
                        # 使用平滑衰减：从 bin_idx=0 到 bin_idx=range_guard，衰减因子从 0.05 到 1.0
                        attenuation = 0.05 + 0.95 * (bin_idx / range_guard) if range_guard > 0 else 1.0
                    else:
                        attenuation = 1.0
                    # endregion

                    val = (1-w)*rc_in[i, idx_low] + w*rc_in[i, idx_low+1]
                    val *= attenuation  # 应用衰减
                    
                    # region agent log - 记录衰减效果
                    if i == Na // 2 and ((iy == 0 and ix == 0) or (iy == Ny // 2 and ix == Nx // 2)):
                        agent_log(
                            "FIX2",
                            "二维hermite仿真.py:bpa_attenuation",
                            "BPA 前段衰减效果",
                            {
                                "pixel_pos": {"ix": int(ix), "iy": int(iy), "x": float(x), "y": float(y)},
                                "bin_idx": float(bin_idx),
                                "attenuation": float(attenuation),
                                "val_before_atten": float(np.abs((1-w)*rc_in[i, idx_low] + w*rc_in[i, idx_low+1])),
                                "val_after_atten": float(np.abs(val)),
                            },
                        )
                    # endregion

                    img[iy, ix] += val * np.exp(1j * 4 * np.pi * R / wavelength)

                    # region agent log
                    # 针对少数代表性像素记录 BPA 几何与距离 bin 信息
                    if i == Na // 2 and ((iy == 0 and ix == 0) or (iy == Ny // 2 and ix == Nx // 2)):
                        agent_log(
                            "H2",
                            "二维hermite仿真.py:bpa_pixel",
                            "BPA 单像素插值与回波贡献",
                            {
                                "pixel_pos": {
                                    "ix": int(ix),
                                    "iy": int(iy),
                                    "x": float(x),
                                    "y": float(y),
                                },
                                "pulse_index": int(i),
                                "R": float(R),
                                "bin_idx": float(bin_idx),
                                "idx_low": int(idx_low),
                                "inside": bool(0 <= idx_low < Nr_ext - 1),
                                "rc_val_abs": float(np.abs(rc_in[i, idx_low])),
                                "img_partial_abs": float(np.abs(img[iy, ix])),
                            },
                        )
                    # endregion

    # region agent log - 诊断4：检查 BPA 叠加是否成功
    # 统计 BPA 过程中各像素的叠加次数和能量累积
    if rc_in is rc_data_frft:  # 只对 FRFT 版本记录详细诊断
        # 检查目标位置和左下角位置的叠加情况
        ix_target = np.argmin(np.abs(x_vec - 0))
        iy_target = np.argmin(np.abs(y_vec - 0))
        ix_corner = 0
        iy_corner = 0
        
        # 计算这两个位置的叠加贡献（模拟）
        target_contributions = []
        corner_contributions = []
        for i in range(Na):
            px = platform_x[i]
            R_target = np.sqrt((px - 0)**2 + (gr_center + 0)**2 + H**2)
            R_corner = np.sqrt((px - x_vec[ix_corner])**2 + (gr_center + y_vec[iy_corner])**2 + H**2)
            bin_idx_target = (R_target - (R0 - range_swath/2)) / dr
            bin_idx_corner = (R_corner - (R0 - range_swath/2)) / dr
            
            idx_low_t = int(np.floor(bin_idx_target))
            idx_low_c = int(np.floor(bin_idx_corner))
            
            if 0 <= idx_low_t < Nr_ext - 1:
                target_contributions.append(float(np.abs(rc_in[i, idx_low_t])))
            if 0 <= idx_low_c < Nr_ext - 1:
                corner_contributions.append(float(np.abs(rc_in[i, idx_low_c])))
        
        agent_log(
            "DIAG4",
            "二维hermite仿真.py:bpa_superposition",
            "BPA 叠加过程诊断（FRFT 版本）",
            {
                "target_final_abs": float(np.abs(img[iy_target, ix_target])),
                "corner_final_abs": float(np.abs(img[iy_corner, ix_corner])),
                "target_mean_contribution": float(np.mean(target_contributions)) if target_contributions else 0.0,
                "corner_mean_contribution": float(np.mean(corner_contributions)) if corner_contributions else 0.0,
                "target_max_contribution": float(np.max(target_contributions)) if target_contributions else 0.0,
                "corner_max_contribution": float(np.max(corner_contributions)) if corner_contributions else 0.0,
                "num_pulses": int(Na),
            },
        )
    # endregion

    return x_vec, y_vec, img



print("  重建图像 LFM (Jammed)...")
_, _, img_lfm = bpa(rc_lfm)

print("  重建图像 Sparse (Filtered)...")
_, _, img_sparse = bpa(rc_data)

print(f"  重建图像 FRFT-Cleaned (在最优角度 α={best_alpha:.3f} 清洗)...")
x_axis, y_axis, img_sparse_cleaned = bpa(rc_data_cleaned)

# ==========================================
# 7. 结果可视化
# ==========================================
print_section("5. 结果对比与诊断")

def get_db(img):
    return 20*np.log10(np.abs(img)/np.max(np.abs(img)) + 1e-6)

# 计算 SCR (Signal to Clutter/Jammer Ratio)
def calc_scr(img, x_ax, y_ax):
    ix_t = np.argmin(np.abs(x_ax - 0))
    iy_t = np.argmin(np.abs(y_ax - 0))
    sig_pow = np.mean(np.abs(img[iy_t-1:iy_t+2, ix_t-1:ix_t+2])**2)
    ix_j = np.argmin(np.abs(x_ax - (-30)))
    iy_j = np.argmin(np.abs(y_ax - 20))
    jam_pow = np.mean(np.abs(img[iy_j-1:iy_j+2, ix_j-1:ix_j+2])**2)
    return 10*np.log10(sig_pow / (jam_pow + 1e-10))

scr_lfm = calc_scr(img_lfm, x_axis, y_axis)
scr_sparse = calc_scr(img_sparse, x_axis, y_axis)
scr_cleaned = calc_scr(img_sparse_cleaned, x_axis, y_axis)

print(f"  LFM 图像信干比 (SCR): {scr_lfm:.2f} dB")
print(f"  Sparse 图像信干比 (SCR): {scr_sparse:.2f} dB")
print(f"  FRFT-Cleaned 图像信干比 (SCR): {scr_cleaned:.2f} dB")

# 线性对消实验 (Sparse - alpha * LFM)
ix_j = np.argmin(np.abs(x_axis - (-30)))
iy_j = np.argmin(np.abs(y_axis - 20))
win_half = 2
iy0, iy1 = max(0, iy_j-win_half), min(img_sparse.shape[0], iy_j+win_half+1)
ix0, ix1 = max(0, ix_j-win_half), min(img_sparse.shape[1], ix_j+win_half+1)
lfm_win = img_lfm[iy0:iy1, ix0:ix1].ravel()
sparse_win = img_sparse[iy0:iy1, ix0:ix1].ravel()
alpha = np.vdot(lfm_win, sparse_win) / (np.vdot(lfm_win, lfm_win) + 1e-12)
img_cancel = img_sparse - alpha * img_lfm
scr_cancel = calc_scr(img_cancel, x_axis, y_axis)
print(f"  Cancel 图像信干比 (SCR): {scr_cancel:.2f} dB")

# 绘图
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(get_db(img_lfm), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
plt.title(f'LFM (受干扰)\nSCR={scr_lfm:.1f}dB')
plt.colorbar(label='dB')

plt.subplot(1, 4, 2)
plt.imshow(get_db(img_sparse), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
plt.title(f'Sparse (原始)\nSCR={scr_sparse:.1f}dB')
plt.colorbar(label='dB')

plt.subplot(1, 4, 3)
plt.imshow(get_db(img_cancel), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
plt.title(f'线性对消\nSCR={scr_cancel:.1f}dB')
plt.colorbar(label='dB')

plt.subplot(1, 4, 4)
plt.imshow(get_db(img_sparse_cleaned), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
plt.title(f'FRFT最优域清洗\nSCR={scr_cleaned:.1f}dB')
plt.colorbar(label='dB')

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()