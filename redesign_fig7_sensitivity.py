#!/usr/bin/env python3
"""
Redesigned Figure 7: Sensitivity of Mixing Coefficient η
Cleaner, publication-ready version
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Data ──
eta = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
head_sr = np.array([0.94, 0.94, 0.94, 0.94, 0.94, 0.94])
tail_sr = np.array([0.52, 0.54, 0.55, 0.56, 0.57, 0.57])
cvar20 = np.array([0.50, 0.53, 0.54, 0.55, 0.56, 0.56])

# ── Styling ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(9, 6))

# ── Plot lines with markers ──
ax.plot(eta, head_sr, 'o-', color='#2E5A88', linewidth=2.5, markersize=10, 
        markerfacecolor='#2E5A88', markeredgecolor='white', markeredgewidth=1.5,
        label='Head SR', zorder=3)

ax.plot(eta, tail_sr, 's-', color='#D4532B', linewidth=2.5, markersize=10,
        markerfacecolor='#D4532B', markeredgecolor='white', markeredgewidth=1.5,
        label='Tail SR', zorder=3)

ax.plot(eta, cvar20, '^-', color='#5A8A4B', linewidth=2.5, markersize=10,
        markerfacecolor='#5A8A4B', markeredgecolor='white', markeredgewidth=1.5,
        label='CVaR@20', zorder=3)

# ── Shaded stable range ──
ax.axvspan(0.3, 0.7, alpha=0.15, color='#5A8A4B', zorder=0)
ax.text(0.5, 0.35, 'Stable Range', ha='center', va='center', fontsize=10, 
        color='#5A8A4B', fontweight='bold', alpha=0.8)

# ── Annotations for key insights ──
# Head SR plateau
ax.annotate('Head SR stable\nacross all η', xy=(0.5, 0.94), xytext=(0.15, 0.88),
            fontsize=9, ha='center', color='#2E5A88',
            arrowprops=dict(arrowstyle='->', color='#2E5A88', lw=1.2))

# Tail SR improvement
ax.annotate('Tail SR improves\nwith higher η', xy=(0.8, 0.57), xytext=(0.85, 0.62),
            fontsize=9, ha='left', color='#D4532B',
            arrowprops=dict(arrowstyle='->', color='#D4532B', lw=1.2))

# ── Axis settings ──
ax.set_xlabel(r'Mixing Coefficient $\eta$', fontsize=13, fontweight='bold')
ax.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.35, 1.0)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(axis='both', labelsize=10)

# ── Title ──
ax.set_title('Sensitivity of Mixing Coefficient', fontsize=14, fontweight='bold', pad=12)

# ── Legend ──
ax.legend(loc='lower right', framealpha=0.95, edgecolor='#CCCCCC', 
          fontsize=10, ncol=1)

# ── Grid ──
ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ── Spine styling ──
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#333333')

plt.tight_layout()
out_path = '/Users/avalok/work/Q-TAIL-MVP/fig7_redesigned_sensitivity.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved to {out_path}')
plt.close()
