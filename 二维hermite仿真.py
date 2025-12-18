# -*- coding: utf-8 -*-
import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import correlate, resample

from scipy.interpolate import interp1d

import warnings



# 忽略除零警告

warnings.filterwarnings('ignore')

plt.style.use('dark_background')



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



# 对照组：LFM受到干扰

print("  生成对照组 (LFM 受干扰)...")

echo_lfm_jam = gen_echo(lfm_wf, use_jam=True)

rc_lfm = np.zeros_like(echo_lfm_jam)

for i in range(Na):

    rc_lfm[i] = correlate(echo_lfm_jam[i], lfm_wf, mode='same')



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

                    val = (1-w)*rc_in[i, idx_low] + w*rc_in[i, idx_low+1]

                    img[iy, ix] += val * np.exp(1j * 4 * np.pi * R / wavelength)

    return x_vec, y_vec, img



print("  重建图像 LFM (Jammed)...")

_, _, img_lfm = bpa(rc_lfm)

print("  重建图像 Sparse (Filtered)...")

x_axis, y_axis, img_sparse = bpa(rc_data)



# ==========================================

# 7. 结果可视化

# ==========================================

print_section("5. 结果对比")



def get_db(img):

    return 20*np.log10(np.abs(img)/np.max(np.abs(img)) + 1e-6)



# 计算 SCR (Signal to Clutter/Jammer Ratio)

# 目标在 (0,0)，干扰在 (-30, 20)

def calc_scr(img, x_ax, y_ax):

    # 目标能量 (中心 3x3 区域)

    ix_t = np.argmin(np.abs(x_ax - 0))

    iy_t = np.argmin(np.abs(y_ax - 0))

    sig_pow = np.mean(np.abs(img[iy_t-1:iy_t+2, ix_t-1:ix_t+2])**2)



    # 干扰能量

    ix_j = np.argmin(np.abs(x_ax - (-30)))

    iy_j = np.argmin(np.abs(y_ax - 20))

    jam_pow = np.mean(np.abs(img[iy_j-1:iy_j+2, ix_j-1:ix_j+2])**2)



    return 10*np.log10(sig_pow / (jam_pow + 1e-10))



scr_lfm = calc_scr(img_lfm, x_axis, y_axis)

scr_sparse = calc_scr(img_sparse, x_axis, y_axis)



print(f"  LFM 图像信干比 (SCR): {scr_lfm:.2f} dB")

print(f"  Sparse 图像信干比 (SCR): {scr_sparse:.2f} dB")

print(f"  >>> 最终提升 (Improvement): {scr_sparse - scr_lfm:.2f} dB <<<")



# 绘图

plt.figure(figsize=(12, 5))



plt.subplot(1, 2, 1)

plt.imshow(get_db(img_lfm), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')

plt.colorbar(label='dB')

plt.title(f'LFM (Jammed)\nSCR = {scr_lfm:.1f} dB')

plt.xlabel('Azimuth (m)')

plt.ylabel('Range (m)')

plt.scatter(0, 0, c='w', marker='o', label='Target')

plt.scatter(-30, 20, c='r', marker='x', label='Jammer')

plt.legend()



plt.subplot(1, 2, 2)

plt.imshow(get_db(img_sparse), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')

plt.colorbar(label='dB')

plt.title(f'Sparse Hermite (Filtered)\nSCR = {scr_sparse:.1f} dB')

plt.xlabel('Azimuth (m)')

plt.scatter(0, 0, c='w', marker='o')

plt.scatter(-30, 20, c='r', marker='x')



plt.tight_layout()

plt.savefig('sar_final_result.png')

print("结果已保存: sar_final_result.png")