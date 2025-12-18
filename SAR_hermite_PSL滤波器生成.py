# -*- coding: utf-8 -*-
"""
SAR杂波抑制与PSF优化 (Clutter Suppression & PSF Optimization)
=============================================================
问题诊断：Sparse Hermite波形的自相关函数旁瓣过高，导致成像出现大量杂波。
解决方案：引入最小二乘(LS)失配滤波器，强制压低旁瓣。

功能：
1. 对比 "标准匹配滤波" vs "LS失配滤波" 的1D距离像。
2. 应用 LS 滤波器进行 2D 成像，消除目标周围的杂波晕。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, windows
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')

# ==========================================
# 1. 基础参数与波形重建
# ==========================================
c = 299792458.0
fc = 5.3e9
B = 80e6
fs = 200e6
Tp = 5e-6
Nr = int(Tp * fs)
t_pulse = np.linspace(-Tp/2, Tp/2, Nr)

# SAR 参数
R0 = 10000.0
H = 5000.0
V = 100.0
PRF = 400.0
Na = 128
range_swath = 200
Nr_ext = Nr * 3 # 扩展长度，用于避免循环卷积混叠
dr = range_swath / Nr_ext

# --- Hermite 基底构建 (同前) ---
max_order = 50
sparse_orders = [10, 18, 27, 36, 45]
x_max_needed = np.sqrt(2 * max_order + 1) + 4.5
sigma = x_max_needed / (Tp / 2)
t_scaled = t_pulse * sigma

hermite_basis = np.zeros((max_order + 1, Nr))
norm_factor = np.pi**(-0.25)
hermite_basis[0, :] = norm_factor * np.exp(-t_scaled**2 / 2)
hermite_basis[1, :] = np.sqrt(2) * t_scaled * hermite_basis[0, :]
for n in range(2, max_order + 1):
    c1 = np.sqrt(2.0 / n)
    c2 = np.sqrt((n - 1.0) / n)
    hermite_basis[n, :] = c1 * t_scaled * hermite_basis[n-1, :] - c2 * hermite_basis[n-2, :]
q, _ = np.linalg.qr(hermite_basis.T)
hermite_basis_ortho = q.T

# --- 构建发射波形 ---
def get_waveform(orders):
    wf = np.zeros(Nr)
    for o in orders:
        wf += hermite_basis_ortho[o, :]
    wf = wf / np.sqrt(np.sum(wf**2))
    return wf.astype(np.complex128) * np.exp(1j * 2 * np.pi * fc * t_pulse)

sparse_wf = get_waveform(sparse_orders)

# ==========================================
# 2. 核心技术：LS滤波器设计 (The Fix)
# ==========================================
def design_ls_filter(waveform, N_filter, lag_output=None):
    """
    设计最小二乘(Least Squares)失配滤波器
    目标：Filter * Waveform ≈ Delta Function
    """
    if lag_output is None:
        lag_output = N_filter // 2

    # 构建卷积矩阵 (Toeplitz matrix)
    # y = C * h
    # C 是 waveform 的卷积矩阵
    # 我们希望 y 接近 delta

    # 频域法求解 (更高效)
    # H(f) = S*(f) / (|S(f)|^2 + lambda)
    # 这实际上是维纳滤波

    N_fft = 4096 # 使用长FFT保证精度
    S_f = np.fft.fft(waveform, n=N_fft)

    # 增加白噪声项(lambda)防止分母为0，同时平滑频谱
    # lambda 越大，旁瓣越低，但主瓣变宽(分辨率降低)
    reg_factor = 0.05 * np.max(np.abs(S_f)**2)

    H_f = np.conj(S_f) / (np.abs(S_f)**2 + reg_factor)

    # 加窗 (频域加窗 = 时域平滑)
    win = np.fft.fft(windows.taylor(N_fft, nbar=4, sll=30), n=N_fft)
    # H_f = H_f * np.abs(win) # 可选：对滤波器加窗进一步压低旁瓣

    h_time = np.fft.ifft(H_f)

    # 截断到实际脉冲长度
    h_time = np.roll(h_time, Nr//2) # 居中
    h_time = h_time[:Nr]

    return h_time

print("设计 LS 滤波器...")
ls_filter = design_ls_filter(sparse_wf, Nr)

# ==========================================
# 3. 1D 诊断对比：为什么你的图会有杂波？
# ==========================================
print("诊断 1D 脉冲响应...")

# 计算自相关 (Standard Matched Filter)
ac_standard = np.abs(correlate(sparse_wf, sparse_wf, mode='same'))
ac_standard = ac_standard / np.max(ac_standard)

# 计算 LS 滤波输出 (Improved)
ac_ls = np.abs(correlate(sparse_wf, ls_filter, mode='same'))
ac_ls = ac_ls / np.max(ac_ls)

# 绘图对比
plt.figure(figsize=(12, 6))
t_axis = np.arange(len(ac_standard)) - len(ac_standard)//2

plt.plot(t_axis, 20*np.log10(ac_standard + 1e-9), 'r', alpha=0.5, label='Standard Matched Filter (Cluttered)')
plt.plot(t_axis, 20*np.log10(ac_ls + 1e-9), 'lime', linewidth=2, label='LS Mismatched Filter (Clean)')
plt.ylim(-60, 0)
plt.title('Why you see clutter: Waveform Sidelobes Comparison')
plt.xlabel('Range Samples')
plt.ylabel('Amplitude (dB)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('debug_psf_comparison.png')
print("✅ 生成 PSF 对比图: debug_psf_comparison.png")

# ==========================================
# 4. 2D 验证：应用到 SAR 成像
# ==========================================
print("验证 2D 成像效果...")

# 模拟一个点目标
eta = np.arange(Na) / PRF - Na/(2*PRF)
platform_x = V * eta
gr_center = np.sqrt(R0**2 - H**2)
target = {'x': 0, 'y': 0, 'rcs': 1.0}

# 生成回波
echo = np.zeros((Na, Nr_ext), dtype=np.complex128)
freq = np.fft.fftfreq(Nr_ext, d=1/fs)
wf_spec = np.fft.fft(sparse_wf, n=Nr_ext)

for i in range(Na):
    px = platform_x[i]
    R = np.sqrt((px - target['x'])**2 + (gr_center + target['y'])**2 + H**2)
    delay = (R - (R0 - range_swath/2)) / (c/2)
    shift_phase = np.exp(-1j * 2 * np.pi * freq * delay)
    echo[i] += target['rcs'] * np.fft.ifft(wf_spec * shift_phase) * np.exp(-1j * 4 * np.pi * R * fc / c)

# 比较两种处理方式
def process_and_image(raw_echo, filter_kernel, title):
    # 距离压缩
    rc = np.zeros_like(raw_echo)
    for i in range(Na):
        rc[i] = correlate(raw_echo[i], filter_kernel, mode='same')

    # BPA 成像 (局部)
    Nx, Ny = 60, 60
    img = np.zeros((Ny, Nx), dtype=np.complex128)
    x_vec = np.linspace(-15, 15, Nx)
    y_vec = np.linspace(-15, 15, Ny)
    rr, xx = np.meshgrid(y_vec, x_vec, indexing='ij')

    for i in range(Na):
        px = platform_x[i]
        dist = np.sqrt((px - xx)**2 + (gr_center + rr)**2 + H**2)
        idx = (dist - (R0 - range_swath/2)) / dr

        # 线性插值
        idx_f = np.floor(idx).astype(int)
        w = idx - idx_f
        mask = (idx_f >= 0) & (idx_f < Nr_ext - 1)

        if np.any(mask):
            val = (1-w[mask])*rc[i, idx_f[mask]] + w[mask]*rc[i, idx_f[mask]+1]
            img[mask] += val * np.exp(1j * 4 * np.pi * dist[mask] * fc / c)

    return img

# 生成图像
print("  Imaging Standard...")
img_std = process_and_image(echo, sparse_wf, "Standard") # 用波形自己做匹配
print("  Imaging LS Filter...")
img_ls = process_and_image(echo, ls_filter, "LS Filter") # 用LS滤波器做匹配

# 显示对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(20*np.log10(np.abs(img_std)/np.max(np.abs(img_std))+1e-6),
           extent=[-15,15,-15,15], cmap='jet', vmin=-35, vmax=0)
plt.title('Standard Processing\n(High Sidelobes/Clutter)')
plt.colorbar(label='dB')

plt.subplot(1, 2, 2)
plt.imshow(20*np.log10(np.abs(img_ls)/np.max(np.abs(img_ls))+1e-6),
           extent=[-15,15,-15,15], cmap='jet', vmin=-35, vmax=0)
plt.title('LS Filtered Processing\n(Suppressed Clutter)')
plt.colorbar(label='dB')

plt.tight_layout()
plt.savefig('clutter_fix_result.png')
print("✅ 生成 2D 效果对比图: clutter_fix_result.png")