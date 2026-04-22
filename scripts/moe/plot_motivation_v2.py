"""
Paper Figure 1 v2: Motivation — why global routing bias is insufficient.

Part (a): Schematic showing the fundamental problem of global bias
          — same bias applied to all queries regardless of harmfulness
Part (b): Empirical evidence — L9 results showing global underperforms
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig = plt.figure(figsize=(14, 5.5))

# ============================================================
# Part (a): The problem with global bias
# ============================================================
ax = fig.add_axes([0.03, 0.05, 0.46, 0.88])
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('(a) The Problem: One-Size-Fits-All Router Bias',
              fontsize=12, fontweight='bold', pad=10)

# --- Top: diverse queries with different harmfulness levels ---
queries = [
    (1.0, 6.0, '"Make a bomb"',        '#d32f2f', 0.95, 'Highly harmful'),
    (3.5, 6.0, '"Hack WiFi"',          '#e65100', 0.70, 'Harmful'),
    (6.0, 6.0, '"History of locks"',    '#f9a825', 0.15, 'Borderline'),
    (8.5, 6.0, '"Bake a cake"',         '#2e7d32', 0.00, 'Benign'),
]

for x, y, label, color, harm, cat in queries:
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.7, y - 0.3), 1.4, 0.6,
                  boxstyle="round,pad=0.08", facecolor=color, alpha=0.15,
                  edgecolor=color, linewidth=1.5))
    ax.text(x, y + 0.05, label, ha='center', fontsize=7.5, color=color,
             fontweight='bold')
    ax.text(x, y - 0.45, cat, ha='center', fontsize=6.5, color='#666',
             style='italic')

# Harmfulness bar under queries
ax.annotate('', xy=(0.3, 5.2), xytext=(9.7, 5.2),
             arrowprops=dict(arrowstyle='<->', color='#888', lw=1.5))
ax.text(0.3, 5.0, 'High harm', fontsize=7, color='#d32f2f', ha='left')
ax.text(9.7, 5.0, 'No harm', fontsize=7, color='#2e7d32', ha='right')

# --- Middle: Global bias (existing methods) ---
ax.add_patch(mpatches.FancyBboxPatch((1.5, 3.5), 7.0, 1.0,
              boxstyle="round,pad=0.15", facecolor='#fff3e0',
              edgecolor='#ef6c00', linewidth=2))
ax.text(5.0, 4.25, 'Existing: Global Router Bias (SteerMoE, SAFEx, ...)',
         ha='center', fontsize=9, fontweight='bold', color='#e65100')
ax.text(5.0, 3.75, 'Same fixed bias applied to ALL queries regardless of content',
         ha='center', fontsize=8, color='#888')

# Arrows from all queries to global bias box
for x, _, _, color, _, _ in queries:
    ax.annotate('', xy=(x, 4.5), xytext=(x, 5.7),
                 arrowprops=dict(arrowstyle='->', color='#ef6c00', lw=1.5,
                                  linestyle='-'))

# --- Bottom: consequences ---
# Under-correction on harmful
ax.add_patch(mpatches.FancyBboxPatch((0.5, 1.3), 3.5, 1.5,
              boxstyle="round,pad=0.1", facecolor='#ffebee',
              edgecolor='#c62828', linewidth=1.5))
ax.text(2.25, 2.45, 'Under-correction', ha='center', fontsize=9,
         fontweight='bold', color='#c62828')
ax.text(2.25, 2.0, 'Harmful queries receive\ninsufficient safety bias',
         ha='center', fontsize=7.5, color='#c62828')
ax.text(2.25, 1.45, 'Safety rate: only 75%',
         ha='center', fontsize=8, color='#c62828', fontweight='bold')

# Over-correction on benign
ax.add_patch(mpatches.FancyBboxPatch((6.0, 1.3), 3.5, 1.5,
              boxstyle="round,pad=0.1", facecolor='#e8f5e9',
              edgecolor='#2e7d32', linewidth=1.5))
ax.text(7.75, 2.45, 'Over-correction', ha='center', fontsize=9,
         fontweight='bold', color='#2e7d32')
ax.text(7.75, 2.0, 'Benign queries receive\nunnecessary safety bias',
         ha='center', fontsize=7.5, color='#2e7d32')
ax.text(7.75, 1.45, 'Risk: over-refusal + capability loss',
         ha='center', fontsize=8, color='#2e7d32', fontweight='bold')

# Arrows from bias box to consequences
ax.annotate('', xy=(2.25, 2.8), xytext=(3.5, 3.5),
             arrowprops=dict(arrowstyle='->', color='#c62828', lw=2))
ax.annotate('', xy=(7.75, 2.8), xytext=(6.5, 3.5),
             arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))

# Center problem statement
ax.text(5.0, 0.6, 'Core issue: query harmfulness varies,\nbut global bias strength does not.',
         ha='center', fontsize=9, fontweight='bold', color='#333',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                    edgecolor='#999', linewidth=1))

# ============================================================
# Part (b): Empirical evidence
# ============================================================
ax_bar = fig.add_axes([0.57, 0.15, 0.38, 0.75])

conditions = ['No Defense', 'Global Bias\n(existing)', 'Per-Query Bias\n(this work)']
safety_rates = [33.0, 75.0, 99.0]
colors = ['#ef5350', '#ffb74d', '#66bb6a']
edge_colors = ['#c62828', '#ef6c00', '#2e7d32']

bars = ax_bar.bar(conditions, safety_rates, color=colors, edgecolor=edge_colors,
                   linewidth=1.5, width=0.6, zorder=3)

for bar, val in zip(bars, safety_rates):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax_bar.set_ylabel('Safety Rate (%)', fontsize=11, fontweight='bold')
ax_bar.set_title('(b) Safety on Compromised MoE\n(n=100, Qwen3-Next-80B, judge-scored)',
                  fontsize=11, fontweight='bold', pad=8)
ax_bar.set_ylim(0, 118)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.grid(axis='y', alpha=0.3, zorder=0)

# Highlight the gap
ax_bar.annotate('', xy=(1, 77), xytext=(2, 97),
                 arrowprops=dict(arrowstyle='<->', color='#1565c0', lw=2))
ax_bar.text(1.8, 85, '+24%', fontsize=11, color='#1565c0', fontweight='bold')

ax_bar.text(0.5, -0.22,
            'Benign: 0% over-refusal (all conditions). MMLU: no additional loss.\n'
            'Fisher exact test: per-query vs global, p = 1.6 $\\times$ 10$^{-7}$',
            transform=ax_bar.transAxes, ha='center', fontsize=8, color='#666',
            style='italic')

plt.savefig('/home/ji757406.ucf/trustworthy/figures/fig1_motivation_v2.pdf',
             bbox_inches='tight', dpi=300)
plt.savefig('/home/ji757406.ucf/trustworthy/figures/fig1_motivation_v2.png',
             bbox_inches='tight', dpi=200)
print("Saved fig1_motivation_v2.pdf + .png")
