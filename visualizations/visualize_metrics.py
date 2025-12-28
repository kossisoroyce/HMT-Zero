#!/usr/bin/env python3
"""
Nurture Layer Metrics Visualization Script

Usage:
    python visualize_metrics.py <metrics_json_file>
    
Example:
    python visualize_metrics.py nurture-metrics-xxx-2025-12-27.json
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np

def load_metrics(filepath):
    """Load metrics JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_stability_plasticity(data, ax):
    """Plot stability and plasticity over time."""
    trajectory = data['trajectory']
    interactions = trajectory['interaction_numbers']
    stability = trajectory['stability']
    plasticity = trajectory['plasticity']
    
    ax.plot(interactions, stability, 'b-', linewidth=2, label='Stability')
    ax.plot(interactions, plasticity, 'r-', linewidth=2, label='Plasticity')
    
    # Mark phase transitions
    phases = trajectory['phases']
    phase_changes = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            phase_changes.append((interactions[i], phases[i]))
    
    for idx, phase in phase_changes:
        ax.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
        ax.annotate(phase, (idx, 0.95), rotation=90, fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Interaction Number')
    ax.set_ylabel('Value')
    ax.set_title('Stability & Plasticity Trajectory')
    ax.legend(loc='center right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

def plot_stance_evolution(data, ax):
    """Plot stance dimension evolution."""
    trajectory = data['trajectory']
    interactions = trajectory['interaction_numbers']
    stance = trajectory['stance']
    
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for i, (dim, values) in enumerate(stance.items()):
        ax.plot(interactions, values, color=colors[i], linewidth=1.5, label=dim, alpha=0.8)
    
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='neutral')
    ax.set_xlabel('Interaction Number')
    ax.set_ylabel('Stance Value')
    ax.set_title('Stance Dimension Evolution')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

def plot_significance_scores(data, ax):
    """Plot significance scores with evaluation markers."""
    trajectory = data['trajectory']
    interactions = trajectory['interaction_numbers']
    scores = trajectory['significance_scores']
    was_evaluated = trajectory['was_evaluated']
    
    # Plot all scores
    ax.bar(interactions, scores, color='lightblue', alpha=0.6, label='Significance')
    
    # Highlight evaluated interactions
    eval_interactions = [interactions[i] for i, e in enumerate(was_evaluated) if e]
    eval_scores = [scores[i] for i, e in enumerate(was_evaluated) if e]
    ax.scatter(eval_interactions, eval_scores, color='green', s=30, zorder=5, label='Evaluated')
    
    # Threshold line (approximate)
    config = data.get('config', {})
    base_threshold = config.get('BASE_THRESHOLD', 0.3)
    ax.axhline(y=base_threshold, color='red', linestyle='--', alpha=0.5, label=f'Base Threshold ({base_threshold})')
    
    ax.set_xlabel('Interaction Number')
    ax.set_ylabel('Significance Score')
    ax.set_title('Significance Scores (green = triggered evaluation)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

def plot_delta_magnitudes(data, ax):
    """Plot delta magnitudes."""
    trajectory = data['trajectory']
    interactions = trajectory['interaction_numbers']
    deltas = trajectory['delta_magnitudes']
    
    ax.bar(interactions, deltas, color='purple', alpha=0.7)
    ax.set_xlabel('Interaction Number')
    ax.set_ylabel('Delta Magnitude')
    ax.set_title('Stance Delta Magnitudes per Interaction')
    ax.grid(True, alpha=0.3)

def plot_final_stance_radar(data, ax):
    """Plot final stance as radar chart."""
    stance = data['instance']['final_stance']
    
    categories = list(stance.keys())
    values = list(stance.values())
    
    # Complete the loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='purple')
    ax.fill(angles, values, alpha=0.25, color='purple')
    
    # Add neutral circle
    neutral = [0.5] * (len(categories) + 1)
    ax.plot(angles, neutral, '--', linewidth=1, color='gray', alpha=0.5)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title('Final Stance (vs neutral 0.5)')

def plot_summary_stats(data, ax):
    """Plot summary statistics as text."""
    instance = data['instance']
    config = data.get('config', {})
    trajectory = data['trajectory']
    
    stats_text = f"""
INSTANCE SUMMARY
────────────────
ID: {instance['id'][:8]}...
Created: {instance['created_at'][:10]}
Duration: {instance['total_interactions']} interactions

FINAL STATE
───────────
Phase: {instance['final_phase']}
Stability: {instance['final_stability']:.1%}
Plasticity: {instance['final_plasticity']:.1%}

EVALUATION STATS
────────────────
Total: {instance['total_interactions']}
Significant: {instance['significant_interactions']}
Eval Rate: {instance['significant_interactions']/max(instance['total_interactions'],1):.1%}

STANCE CHANGES (from 0.5)
─────────────────────────"""
    
    for dim, val in instance['final_stance'].items():
        change = val - 0.5
        arrow = "↑" if change > 0.05 else "↓" if change < -0.05 else "→"
        stats_text += f"\n{dim}: {val:.2f} {arrow}"
    
    stats_text += f"""

KEY TRAITS
──────────
{chr(10).join('• ' + t for t in instance['final_environment'].get('key_traits', [])[:6])}
"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_metrics.py <metrics_json_file>")
        print("Example: python visualize_metrics.py nurture-metrics-xxx-2025-12-27.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Loading metrics from: {filepath}")
    
    data = load_metrics(filepath)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Nurture Layer Test Results', fontsize=14, fontweight='bold')
    
    # Layout: 3x2 grid
    ax1 = fig.add_subplot(2, 3, 1)  # Stability/Plasticity
    ax2 = fig.add_subplot(2, 3, 2)  # Stance evolution
    ax3 = fig.add_subplot(2, 3, 3)  # Summary stats
    ax4 = fig.add_subplot(2, 3, 4)  # Significance scores
    ax5 = fig.add_subplot(2, 3, 5)  # Delta magnitudes
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')  # Radar chart
    
    plot_stability_plasticity(data, ax1)
    plot_stance_evolution(data, ax2)
    plot_summary_stats(data, ax3)
    plot_significance_scores(data, ax4)
    plot_delta_magnitudes(data, ax5)
    plot_final_stance_radar(data, ax6)
    
    plt.tight_layout()
    
    # Save figure
    output_file = filepath.replace('.json', '_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    
    # Also show if running interactively
    plt.show()

if __name__ == '__main__':
    main()
