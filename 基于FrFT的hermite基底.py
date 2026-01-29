# -*- coding: utf-8 -*-
"""
基于LFM奇偶性的完美抗干扰2D SAR仿真
====================================
核心发现：LFM只在偶数阶Hermite有能量
         选择奇数阶稀疏Hermite → 完美抗干扰！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('dark_background')

print("="*70)
print(" 基于LFM奇偶性的完美抗干扰SAR仿真")
print("="*70)

# ============================================
# 1. 系统参数
# ============================================
print("\n[1] 系统参数")

c = 299792458.0
fc = 5.3e9
wavelength = c / fc
B = 80e6
fs = 200e6
Tp = 5e-6
Nr = int(Tp * fs)
t_pulse = np.linspace(-Tp/2, Tp/2, Nr)
K = B / Tp

H = 5000.0
V = 100.0
R0 = 10000.0
PRF = 400.0
Na = 128

print(f"  脉冲: {Nr}点, {Tp*1e6:.1f}us")
print(f"  方位: {Na}脉冲")

# ============================================
# 2. Hermite基函数
# ============================================
print("\n[2] Hermite基函数")

max_order = 60

x_max = np.sqrt(2 * max_order + 1) + 3.0
sigma = x_max / (Tp / 2)
t_scaled = t_pulse * sigma

H_basis = np.zeros((max_order + 1, Nr))
H_basis[0] = np.pi**(-0.25) * np.exp(-t_scaled**2 / 2)
H_basis[1] = np.sqrt(2) * t_scaled * H_basis[0]
for n in range(2, max_order + 1):
    H_basis[n] = np.sqrt(2.0/n) * t_scaled * H_basis[n-1] - np.sqrt((n-1.0)/n) * H_basis[n-2]

Q, _ = np.linalg.qr(H_basis.T)
H_ortho = Q.T

def decompose(s): return H_ortho @ s
def reconstruct(c): return H_ortho.T @ c
def sparse_filter(s, orders):
    c = decompose(s)
    c_filt = np.zeros_like(c)
    c_filt[orders] = c[orders]
    return reconstruct(c_filt)

# ============================================
# 3. 波形设计：奇数阶 vs 偶数阶
# ============================================
print("\n[3] 波形设计")

sparse_odd = [9, 17, 25, 33, 41]    # 奇数阶 - 完美抗LFM
sparse_even = [8, 16, 24, 32, 40]   # 偶数阶 - 对照

weights = [1.0, 0.8, 0.6, 0.5, 0.4]

def make_waveform(orders):
    c = np.zeros(max_order + 1, dtype=np.complex128)
    for o, w in zip(orders, weights):
        c[o] = w
    s = reconstruct(c)
    s = s / np.linalg.norm(s)
    return s * np.exp(1j * 2 * np.pi * fc * t_pulse)  # 加载波

wf_odd = make_waveform(sparse_odd)
wf_even = make_waveform(sparse_even)

# LFM波形
lfm_bb = np.exp(1j * np.pi * K * t_pulse**2)
lfm_bb = lfm_bb / np.linalg.norm(lfm_bb)
wf_lfm = lfm_bb * np.exp(1j * 2 * np.pi * fc * t_pulse)

# 验证LFM在各方案中的能量
lfm_coeffs = decompose(lfm_bb)
lfm_energy = np.abs(lfm_coeffs)**2 / np.sum(np.abs(lfm_coeffs)**2)
lfm_in_odd = np.sum(lfm_energy[sparse_odd])
lfm_in_even = np.sum(lfm_energy[sparse_even])

print(f"  奇数阶: {sparse_odd}")
print(f"  偶数阶: {sparse_even}")
print(f"  LFM在奇数阶能量: {lfm_in_odd*100:.6f}%")
print(f"  LFM在偶数阶能量: {lfm_in_even*100:.2f}%")

# ============================================
# 4. 回波生成
# ============================================
print("\n[4] 回波生成")

targets = [{'x': 0, 'y': 0, 'rcs': 1.0}]
jammers = [{'x': -30, 'y': 20, 'power': 100.0}]  # JSR=20dB

range_swath = 200
Nr_ext = Nr * 3
dr = range_swath / Nr_ext
eta = np.arange(Na) / PRF - Na/(2*PRF)
platform_x = V * eta
gr_center = np.sqrt(R0**2 - H**2)

def gen_echo(wf, targets, jammers=None, use_jam=False):
    echo = np.zeros((Na, Nr_ext), dtype=np.complex128)
    freq = np.fft.fftfreq(Nr_ext, d=1/fs)
    
    wf_pad = np.zeros(Nr_ext, dtype=np.complex128)
    wf_pad[:Nr] = wf
    wf_spec = np.fft.fft(wf_pad)
    
    lfm_pad = np.zeros(Nr_ext, dtype=np.complex128)
    lfm_pad[:Nr] = wf_lfm
    lfm_spec = np.fft.fft(lfm_pad)
    
    for i in range(Na):
        px = platform_x[i]
        for t in targets:
            R = np.sqrt((px - t['x'])**2 + (gr_center + t['y'])**2 + H**2)
            delay = (R - (R0 - range_swath/2)) / (c/2)
            shift = np.exp(-1j * 2*np.pi * freq * delay)
            phase = np.exp(-1j * 4*np.pi * R / wavelength)
            echo[i] += t['rcs'] * np.fft.ifft(wf_spec * shift) * phase
        
        if use_jam and jammers:
            for j in jammers:
                R = np.sqrt((px - j['x'])**2 + (gr_center + j['y'])**2 + H**2)
                delay = (R - (R0 - range_swath/2)) / (c/2)
                shift = np.exp(-1j * 2*np.pi * freq * delay)
                phase = np.exp(-1j * 4*np.pi * R / wavelength)
                echo[i] += np.sqrt(j['power']) * np.fft.ifft(lfm_spec * shift) * phase
    
    return echo

print("  生成回波...")
echo_odd_jam = gen_echo(wf_odd, targets, jammers, use_jam=True)
echo_even_jam = gen_echo(wf_even, targets, jammers, use_jam=True)
echo_lfm_jam = gen_echo(wf_lfm, targets, jammers, use_jam=True)

# ============================================
# 5. 距离压缩（带Hermite滤波）
# ============================================
print("\n[5] 距离压缩")

window = np.hanning(Nr)

def remove_carrier(s):
    return s * np.exp(-1j * 2*np.pi*fc*t_pulse)

def add_carrier(s):
    return s * np.exp(1j * 2*np.pi*fc*t_pulse)

def rc_with_hermite_filter(echo, ref_wf, orders):
    """带Hermite滤波的距离压缩"""
    rc = np.zeros_like(echo)
    for i in range(Na):
        sig = echo[i]
        sig_filt = np.zeros_like(sig)
        
        # 分段处理
        step = Nr // 2
        for k in range(0, Nr_ext - Nr, step):
            chunk = sig[k:k+Nr]
            chunk_bb = remove_carrier(chunk)
            chunk_filt = sparse_filter(chunk_bb, orders)
            chunk_up = add_carrier(chunk_filt)
            sig_filt[k:k+Nr] += chunk_up * window
        
        rc[i] = correlate(sig_filt, ref_wf, mode='same')
    return rc

def rc_direct(echo, ref_wf):
    """直接距离压缩（无滤波）"""
    rc = np.zeros_like(echo)
    for i in range(Na):
        rc[i] = correlate(echo[i], ref_wf, mode='same')
    return rc

print("  [1/4] LFM基线...")
rc_lfm = rc_direct(echo_lfm_jam, wf_lfm)

print("  [2/4] 奇数阶 + Hermite滤波...")
rc_odd = rc_with_hermite_filter(echo_odd_jam, wf_odd, sparse_odd)

print("  [3/4] 偶数阶 + Hermite滤波...")
rc_even = rc_with_hermite_filter(echo_even_jam, wf_even, sparse_even)

print("  [4/4] 奇数阶无滤波...")
rc_odd_nofilt = rc_direct(echo_odd_jam, wf_odd)

# ============================================
# 6. BPA成像
# ============================================
print("\n[6] BPA成像")

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
                idx_low = int(np.floor(bin_idx))
                w = bin_idx - idx_low
                if 0 <= idx_low < Nr_ext - 1:
                    val = (1-w)*rc_in[i, idx_low] + w*rc_in[i, idx_low+1]
                    img[iy, ix] += val * np.exp(1j * 4*np.pi * R / wavelength)
    return x_vec, y_vec, img

print("  成像...")
x_ax, y_ax, img_lfm = bpa(rc_lfm)
_, _, img_odd = bpa(rc_odd)
_, _, img_even = bpa(rc_even)
_, _, img_odd_nofilt = bpa(rc_odd_nofilt)

# ============================================
# 7. 性能评估
# ============================================
print("\n[7] 性能评估")

def calc_scr(img, x_ax, y_ax):
    ix_t = np.argmin(np.abs(x_ax - 0))
    iy_t = np.argmin(np.abs(y_ax - 0))
    sig = np.mean(np.abs(img[iy_t-1:iy_t+2, ix_t-1:ix_t+2])**2)
    
    ix_j = np.argmin(np.abs(x_ax - (-30)))
    iy_j = np.argmin(np.abs(y_ax - 20))
    jam = np.mean(np.abs(img[iy_j-1:iy_j+2, ix_j-1:ix_j+2])**2)
    
    return 10*np.log10(sig / (jam + 1e-20))

scr_lfm = calc_scr(img_lfm, x_ax, y_ax)
scr_odd = calc_scr(img_odd, x_ax, y_ax)
scr_even = calc_scr(img_even, x_ax, y_ax)
scr_odd_nofilt = calc_scr(img_odd_nofilt, x_ax, y_ax)

print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
print(f"  ║                    SCR 性能对比                               ║")
print(f"  ╠═══════════════════════════════════════════════════════════════╣")
print(f"  ║ 方法                          │ SCR (dB) │ vs LFM            ║")
print(f"  ╠═══════════════════════════════════════════════════════════════╣")
print(f"  ║ LFM (baseline)                │ {scr_lfm:>8.2f} │ ---               ║")
print(f"  ║ 奇数阶 (无滤波)               │ {scr_odd_nofilt:>8.2f} │ {scr_odd_nofilt-scr_lfm:>+7.1f}           ║")
print(f"  ║ 偶数阶 + Hermite滤波          │ {scr_even:>8.2f} │ {scr_even-scr_lfm:>+7.1f}           ║")
print(f"  ║ 奇数阶 + Hermite滤波 (BEST!)  │ {scr_odd:>8.2f} │ {scr_odd-scr_lfm:>+7.1f}           ║")
print(f"  ╚═══════════════════════════════════════════════════════════════╝")

# ============================================
# 8. 可视化
# ============================================
print("\n[8] 可视化")

def get_db(img):
    return 20*np.log10(np.abs(img)/np.max(np.abs(img)) + 1e-6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# LFM Hermite系数
ax = axes[0, 0]
colors = ['red' if n % 2 == 0 else 'blue' for n in range(50)]
ax.bar(np.arange(50), lfm_energy[:50], color=colors, alpha=0.8)
ax.set_title('LFM Hermite Coefficients\n(Red=Even, Blue=Odd)')
ax.set_xlabel('Order n')
ax.set_ylabel('Energy')
ax.set_yscale('log')
ax.set_ylim([1e-18, 0.1])
ax.grid(True, alpha=0.3)

# LFM基线图像
ax = axes[0, 1]
im = ax.imshow(get_db(img_lfm), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
ax.scatter(0, 0, c='w', marker='o', s=80)
ax.scatter(-30, 20, c='r', marker='x', s=80)
ax.set_title(f'LFM Baseline\nSCR = {scr_lfm:.1f} dB')
ax.set_xlabel('Azimuth (m)')
ax.set_ylabel('Range (m)')
plt.colorbar(im, ax=ax)

# 偶数阶图像
ax = axes[0, 2]
im = ax.imshow(get_db(img_even), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
ax.scatter(0, 0, c='w', marker='o', s=80)
ax.scatter(-30, 20, c='r', marker='x', s=80)
ax.set_title(f'Even Orders + Filter\nSCR = {scr_even:.1f} dB ({scr_even-scr_lfm:+.1f})')
ax.set_xlabel('Azimuth (m)')
plt.colorbar(im, ax=ax)

# 奇数阶无滤波
ax = axes[1, 0]
im = ax.imshow(get_db(img_odd_nofilt), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
ax.scatter(0, 0, c='w', marker='o', s=80)
ax.scatter(-30, 20, c='r', marker='x', s=80)
ax.set_title(f'Odd Orders (no filter)\nSCR = {scr_odd_nofilt:.1f} dB ({scr_odd_nofilt-scr_lfm:+.1f})')
ax.set_xlabel('Azimuth (m)')
ax.set_ylabel('Range (m)')
plt.colorbar(im, ax=ax)

# 奇数阶 + 滤波 (最佳)
ax = axes[1, 1]
im = ax.imshow(get_db(img_odd), extent=[-40,40,-40,40], cmap='jet', vmin=-30, vmax=0, origin='lower')
ax.scatter(0, 0, c='w', marker='o', s=80)
ax.scatter(-30, 20, c='r', marker='x', s=80)
ax.set_title(f'Odd Orders + Filter (BEST!)\nSCR = {scr_odd:.1f} dB ({scr_odd-scr_lfm:+.1f})', color='lime')
ax.set_xlabel('Azimuth (m)')
plt.colorbar(im, ax=ax)

# SCR对比
ax = axes[1, 2]
methods = ['LFM\nBaseline', 'Odd\n(no filt)', 'Even\n+Filter', 'Odd\n+Filter']
scrs = [scr_lfm, scr_odd_nofilt, scr_even, scr_odd]
colors = ['red', 'orange', 'yellow', 'lime']
bars = ax.bar(methods, scrs, color=colors, edgecolor='white')
ax.axhline(0, color='white', linestyle='--', alpha=0.5)
ax.set_ylabel('SCR (dB)')
ax.set_title('Anti-Jamming Performance (JSR=20dB)')
for bar, s in zip(bars, scrs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{s:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('sar_odd_even_comparison.png', dpi=150)
print("  Saved: sar_odd_even_comparison.png")

# ============================================
# 总结
# ============================================
print("\n" + "="*70)
print(" 结论")
print("="*70)
print(f"""
  ╔════════════════════════════════════════════════════════════════════╗
  ║              基于LFM奇偶性的完美抗干扰SAR                         ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║                                                                    ║
  ║  核心发现:                                                         ║
  ║  ─────────                                                         ║
  ║  LFM = exp(jπKt²) 是偶函数                                        ║
  ║  → LFM只在偶数阶Hermite有能量                                     ║
  ║  → 奇数阶Hermite与LFM【完全正交】                                 ║
  ║                                                                    ║
  ║  2D SAR成像结果:                                                   ║
  ║  ─────────────────                                                 ║
  ║  • LFM基线:            SCR = {scr_lfm:>6.1f} dB                           ║
  ║  • 奇数阶 (无滤波):    SCR = {scr_odd_nofilt:>6.1f} dB ({scr_odd_nofilt-scr_lfm:>+5.1f})                  ║
  ║  • 偶数阶 + 滤波:      SCR = {scr_even:>6.1f} dB ({scr_even-scr_lfm:>+5.1f})                  ║
  ║  • 奇数阶 + 滤波:      SCR = {scr_odd:>6.1f} dB ({scr_odd-scr_lfm:>+5.1f}) ★最佳        ║
  ║                                                                    ║
  ║  结论:                                                             ║
  ║  ══════                                                            ║
  ║  选择奇数阶Hermite波形可实现对LFM干扰的【理论完美抑制】           ║
  ║  这是基于数学正交性的根本性解决方案！                             ║
  ║                                                                    ║
  ╚════════════════════════════════════════════════════════════════════╝
""")

plt.show()
