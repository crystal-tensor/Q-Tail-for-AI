#!/usr/bin/env python3
"""
Redesigned Figure: Per-Task Sample Allocation & Success Rate under PT-rank
- Dual Y-axis with clear visual separation
- PT-rank vs Uniform comparison
- Head/Medium/Tail task grouping
- Professional color scheme
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data (from paper) ──
tasks = ['reach', 'push', 'pick-place', 'door-open', 
         'drawer-close', 'button-press', 'peg-insert',
         'window-open', 'sweep', 'basketball']

# Task categories
categories = {
    'reach': 'Head', 'push': 'Head', 'pick-place': 'Head', 'door-open': 'Head',
    'drawer-close': 'Medium', 'button-press': 'Medium', 'peg-insert': 'Medium',
    'window-open': 'Tail', 'sweep': 'Tail', 'basketball': 'Tail'
}

# PT-rank sample allocation (%) — from paper Fig.8 data
pt_rank_samples = [5.8, 6.0, 7.5, 7.8, 9.5, 9.4, 11.0, 12.5, 13.5, 16.5]

# Uniform baseline (%)
uniform_samples = [10.0] * 10

# Success rates (%) — from paper Table 1 + per-task breakdown
pt_rank_sr = [0.96, 0.95, 0.94, 0.93, 0.88, 0.85, 0.78, 0.62, 0.55, 0.48]
uniform_sr = [0.95, 0.94, 0.93, 0.92, 0.82, 0.79, 0.70, 0.50, 0.44, 0.38]

# ── Styling ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
})

fig, ax1 = plt.subplots(figsize=(14, 7.5))

x = np.arange(len(tasks))
bar_width = 0.35

# ── Bar chart: Sample Allocation (left Y-axis) ──
bars_pt = ax1.bar(x - bar_width/2, pt_rank_samples, bar_width, 
                   label='PT-rank allocation', color='#2E5A88', edgecolor='white', linewidth=0.6, zorder=3)
bars_uni = ax1.bar(x + bar_width/2, uniform_samples, bar_width,
                    label='Uniform allocation', color='#C0C0C0', edgecolor='white', linewidth=0.6, 
                    hatch='///', alpha=0.75, zorder=3)

ax1.set_ylabel('Sample Allocation (%)', fontsize=15, fontweight='bold', color='#2E5A88')
ax1.set_ylim(0, 20)
ax1.set_yticks([0, 5, 10, 15, 20])
ax1.tick_params(axis='y', labelcolor='#2E5A88', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# Add value labels on bars
for bar, val in zip(bars_pt, pt_rank_samples):
    if val >= 10:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E5A88')
    else:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='#555555')

# ── Line chart: Success Rate (right Y-axis) ──
ax2 = ax1.twinx()
line_pt = ax2.plot(x, pt_rank_sr, 'o-', color='#D4532B', linewidth=2.8, markersize=10,
                    label='PT-rank success rate', zorder=5, markeredgecolor='white', markeredgewidth=1.5)
line_uni = ax2.plot(x, uniform_sr, 's--', color='#7A7A7A', linewidth=2.0, markersize=7,
                     label='Uniform success rate', zorder=4, markeredgecolor='white', markeredgewidth=1)

ax2.set_ylabel('Success Rate', fontsize=15, fontweight='bold', color='#D4532B')
ax2.set_ylim(0.30, 1.05)
ax2.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.tick_params(axis='y', labelcolor='#D4532B', labelsize=12)

# Add SR value annotations on line points
for i, (sr_pt, sr_u) in enumerate(zip(pt_rank_sr, uniform_sr)):
    # Only annotate tail tasks and key transitions to avoid clutter
    if categories[tasks[i]] in ('Medium', 'Tail'):
        ax2.annotate(f'{sr_pt:.0%}', (x[i], sr_pt), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color='#D4532B')

# ── Category background shading ──
head_end = 4   # first 4 tasks are Head
med_end = 7    # next 3 are Medium
tail_end = 10  # last 3 are Tail

ax1.axvspan(-0.55, head_end - 0.45, alpha=0.08, color='#2E86AB', zorder=0)
ax1.axvspan(head_end - 0.55, med_end - 0.45, alpha=0.08, color='#F18F01', zorder=0)
ax1.axvspan(med_end - 0.55, tail_end - 0.45, alpha=0.08, color='#C73E1D', zorder=0)

# Category labels at top
ax1.text(1.5, 19.5, 'HEAD', ha='center', va='top', fontsize=14, fontweight='bold', 
         color='#2E86AB', alpha=0.85)
ax1.text(5.5, 19.5, 'MEDIUM', ha='center', va='top', fontsize=14, fontweight='bold',
         color='#B07A0C', alpha=0.85)
ax1.text(8.5, 19.5, 'TAIL', ha='center', va='top', fontsize=14, fontweight='bold',
         color='#C73E1D', alpha=0.85)

# Vertical separators between categories
for sep_x in [head_end - 0.45, med_end - 0.45]:
    ax1.axvline(x=sep_x, color='#333333', linestyle=':', linewidth=1.0, alpha=0.5, zorder=1)

# ── X-axis ──
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, rotation=30, ha='right', fontsize=12)
ax1.set_xlabel('Meta-World MT10 Tasks', fontsize=14, fontweight='bold', labelpad=10)

# ── Legend (combined from both axes) ──
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
all_handles = handles1 + handles2
all_labels = labels1 + labels2
legend = ax1.legend(all_handles, all_labels, loc='upper left', framealpha=0.95, 
                    edgecolor='#CCCCCC', fontsize=11, ncol=2)

# ── Annotation callout box ──
textstr = f'PT-rank Tail SR: {np.mean(pt_rank_sr[7:])*100:.1f}%\nUniform Tail SR: {np.mean(uniform_sr[7:])*100:.1f}%\nΔ = +{np.mean(pt_rank_sr[7:])*100 - np.mean(uniform_sr[7:])*100:.1f}%'
props = dict(boxstyle='round,pad=0.5', facecolor='#FFF8E7', edgecolor='#D4532B', alpha=0.92)
ax2.text(0.97, 0.42, textstr, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right', bbox=props,
         family='monospace')

# ── Title ──
plt.title('Per-Task Sample Allocation & Success Rate under PT-rank\n(Meta-World MT10, 100k steps × 5 seeds)', 
          fontsize=16, fontweight='bold', pad=18)

# Grid on left axis only
ax1.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
ax1.set_axisbelow(True)

plt.tight_layout()
out_path = '/Users/avalok/work/Q-TAIL-MVP/fig9_redesigned_pertask_allocation.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved to {out_path}')
plt.close()
