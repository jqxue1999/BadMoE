"""
Paper Figure 1: Motivation figure for per-query safety-aware MoE routing.

Part (a): Schematic — global vs per-query bias on two different queries
Part (b): Bar chart — L9 main result (3 bars, same attack setting)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig = plt.figure(figsize=(14, 5))

# ============================================================
# Part (a): Schematic — left half
# ============================================================
ax_schema = fig.add_axes([0.02, 0.05, 0.48, 0.90])
ax_schema.set_xlim(0, 10)
ax_schema.set_ylim(0, 6)
ax_schema.axis('off')
ax_schema.set_title('(a) Global vs Per-Query Safety Bias', fontsize=13, fontweight='bold', pad=10)

# --- Global bias (left column) ---
ax_schema.text(2.5, 5.7, 'Global Bias', ha='center', fontsize=11, fontweight='bold', color='#555')

# Query boxes
ax_schema.add_patch(mpatches.FancyBboxPatch((1.0, 4.6), 3.0, 0.7, boxstyle="round,pad=0.1",
                                              facecolor='#ffcccc', edgecolor='#cc0000', linewidth=1.5))
ax_schema.text(2.5, 4.95, '"How to make bomb?"', ha='center', fontsize=8, color='#cc0000', fontweight='bold')

ax_schema.add_patch(mpatches.FancyBboxPatch((1.0, 3.6), 3.0, 0.7, boxstyle="round,pad=0.1",
                                              facecolor='#ccffcc', edgecolor='#00aa00', linewidth=1.5))
ax_schema.text(2.5, 3.95, '"How to bake cake?"', ha='center', fontsize=8, color='#00aa00', fontweight='bold')

# Arrows to router
ax_schema.annotate('', xy=(2.5, 3.0), xytext=(2.5, 3.6),
                    arrowprops=dict(arrowstyle='->', color='#ff8800', lw=2.5))
ax_schema.annotate('', xy=(2.5, 3.0), xytext=(2.5, 4.6),
                    arrowprops=dict(arrowstyle='->', color='#ff8800', lw=2.5))

# Router box
ax_schema.add_patch(mpatches.FancyBboxPatch((1.2, 2.3), 2.6, 0.7, boxstyle="round,pad=0.1",
                                              facecolor='#ffe0b2', edgecolor='#ff8800', linewidth=2))
ax_schema.text(2.5, 2.65, 'Router + SAME bias', ha='center', fontsize=8, fontweight='bold', color='#cc6600')

# Result
ax_schema.text(2.5, 1.6, 'Same correction\nfor both queries', ha='center', fontsize=8,
                color='#888', style='italic')
ax_schema.text(2.5, 0.8, 'Harmful: under-corrected\nBenign: over-corrected', ha='center',
                fontsize=8, color='#cc0000', fontweight='bold')

# --- Per-query bias (right column) ---
ax_schema.text(7.5, 5.7, 'Per-Query Bias (Ours)', ha='center', fontsize=11, fontweight='bold', color='#555')

# Harmful query
ax_schema.add_patch(mpatches.FancyBboxPatch((6.0, 4.6), 3.0, 0.7, boxstyle="round,pad=0.1",
                                              facecolor='#ffcccc', edgecolor='#cc0000', linewidth=1.5))
ax_schema.text(7.5, 4.95, '"How to make bomb?"', ha='center', fontsize=8, color='#cc0000', fontweight='bold')

# Benign query
ax_schema.add_patch(mpatches.FancyBboxPatch((6.0, 3.6), 3.0, 0.7, boxstyle="round,pad=0.1",
                                              facecolor='#ccffcc', edgecolor='#00aa00', linewidth=1.5))
ax_schema.text(7.5, 3.95, '"How to bake cake?"', ha='center', fontsize=8, color='#00aa00', fontweight='bold')

# Lambda probe
ax_schema.add_patch(mpatches.FancyBboxPatch((5.5, 3.2), 1.5, 0.3, boxstyle="round,pad=0.05",
                                              facecolor='#e0e0ff', edgecolor='#4444cc', linewidth=1))
ax_schema.text(6.25, 3.35, 'λ=2.1', ha='center', fontsize=7, color='#4444cc', fontweight='bold')

ax_schema.add_patch(mpatches.FancyBboxPatch((8.0, 3.2), 1.5, 0.3, boxstyle="round,pad=0.05",
                                              facecolor='#e0e0ff', edgecolor='#4444cc', linewidth=1))
ax_schema.text(8.75, 3.35, 'λ=0.0', ha='center', fontsize=7, color='#4444cc', fontweight='bold')

# Arrows from queries to lambda
ax_schema.annotate('', xy=(6.25, 3.5), xytext=(7.0, 4.6),
                    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.5))
ax_schema.annotate('', xy=(8.75, 3.5), xytext=(8.0, 3.6),
                    arrowprops=dict(arrowstyle='->', color='#00aa00', lw=1.5))

# Arrows to router
ax_schema.annotate('', xy=(6.8, 2.95), xytext=(6.25, 3.2),
                    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=2.5))
ax_schema.annotate('', xy=(8.2, 2.95), xytext=(8.75, 3.2),
                    arrowprops=dict(arrowstyle='->', color='#00aa00', lw=1.0, linestyle='--'))

# Router box with adaptive
ax_schema.add_patch(mpatches.FancyBboxPatch((6.0, 2.2), 1.5, 0.75, boxstyle="round,pad=0.1",
                                              facecolor='#ffcdd2', edgecolor='#cc0000', linewidth=2))
ax_schema.text(6.75, 2.58, 'Router\n+STRONG bias', ha='center', fontsize=7, fontweight='bold', color='#cc0000')

ax_schema.add_patch(mpatches.FancyBboxPatch((8.0, 2.2), 1.5, 0.75, boxstyle="round,pad=0.1",
                                              facecolor='#e8f5e9', edgecolor='#00aa00', linewidth=2))
ax_schema.text(8.75, 2.58, 'Router\nNO bias', ha='center', fontsize=7, fontweight='bold', color='#00aa00')

# Results
ax_schema.text(6.75, 1.5, 'Safety expert\nactivated → Refuse', ha='center', fontsize=8,
                color='#0066cc', fontweight='bold')
ax_schema.text(8.75, 1.5, 'Normal routing\n→ Helpful answer', ha='center', fontsize=8,
                color='#00aa00', fontweight='bold')

# Divider
ax_schema.plot([5.0, 5.0], [0.5, 5.5], color='#cccccc', linewidth=1, linestyle='--')

# ============================================================
# Part (b): Bar chart — right half
# ============================================================
ax_bar = fig.add_axes([0.58, 0.15, 0.38, 0.75])

conditions = ['No Defense', 'Global Bias\n(λ=1)', 'Per-Query Bias\n(Ours)']
safety_rates = [33.0, 75.0, 99.0]
colors = ['#ef5350', '#ffb74d', '#66bb6a']
edge_colors = ['#c62828', '#ef6c00', '#2e7d32']

bars = ax_bar.bar(conditions, safety_rates, color=colors, edgecolor=edge_colors,
                   linewidth=1.5, width=0.6, zorder=3)

# Value labels
for bar, val in zip(bars, safety_rates):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax_bar.set_ylabel('Safety Rate (%)', fontsize=11, fontweight='bold')
ax_bar.set_title('(b) Safety Restoration on Compromised MoE\n(n=100, Qwen3-Next-80B, judge-scored)',
                  fontsize=11, fontweight='bold', pad=8)
ax_bar.set_ylim(0, 115)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.grid(axis='y', alpha=0.3, zorder=0)

# Annotation
ax_bar.text(0.5, -0.18, 'Benign over-refusal: 0% all conditions. MMLU: no additional loss.\n'
            'Fisher exact: per-query vs global p = 1.6×10⁻⁷',
            transform=ax_bar.transAxes, ha='center', fontsize=8, color='#666', style='italic')

plt.savefig('/home/ji757406.ucf/trustworthy/figures/fig1_motivation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('/home/ji757406.ucf/trustworthy/figures/fig1_motivation.png', bbox_inches='tight', dpi=200)
print("Saved fig1_motivation.pdf + .png")
