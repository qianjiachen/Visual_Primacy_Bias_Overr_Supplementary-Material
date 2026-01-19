"""Defense evaluation visualization utilities.

Provides visualization for defense effectiveness analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple


def plot_defense_comparison(
    defense_results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Defense Effectiveness Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot comparison of different defenses.
    
    Args:
        defense_results: Dict mapping defense name to metrics dict
        metrics: List of metrics to plot
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['asr_reduction', 'clean_accuracy', 'latency']
    
    defenses = list(defense_results.keys())
    x = np.arange(len(defenses))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        values = [defense_results[d].get(metric, 0) for d in defenses]
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric, color=colors[i])
    
    ax.set_xlabel('Defense')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(defenses, rotation=45, ha='right')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_asr_by_defense_level(
    asr_values: Dict[str, List[float]],
    defense_levels: List[str],
    title: str = "ASR by Defense Level",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot ASR across different defense levels.
    
    Args:
        asr_values: Dict mapping attack type to ASR values per defense level
        defense_levels: List of defense level names
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(defense_levels))
    width = 0.8 / len(asr_values)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(asr_values)))
    
    for i, (attack_type, values) in enumerate(asr_values.items()):
        offset = (i - len(asr_values)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=attack_type, color=colors[i])
    
    ax.set_xlabel('Defense Level')
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(defense_levels)
    ax.legend()
    ax.set_ylim(0, 100)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_defense_tradeoff(
    defense_results: Dict[str, Tuple[float, float]],
    x_metric: str = "ASR Reduction (%)",
    y_metric: str = "Clean Accuracy (%)",
    title: str = "Defense Trade-off Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot trade-off between defense effectiveness and clean performance.
    
    Args:
        defense_results: Dict mapping defense name to (x_value, y_value)
        x_metric: Label for x-axis
        y_metric: Label for y-axis
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    defenses = list(defense_results.keys())
    x_values = [defense_results[d][0] for d in defenses]
    y_values = [defense_results[d][1] for d in defenses]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(defenses)))
    
    for i, defense in enumerate(defenses):
        ax.scatter(x_values[i], y_values[i], s=150, c=[colors[i]], 
                   label=defense, edgecolors='black', linewidths=1)
    
    # Add Pareto frontier
    pareto_x, pareto_y = _compute_pareto_frontier(x_values, y_values)
    ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, label='Pareto Frontier')
    
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _compute_pareto_frontier(x: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    """Compute Pareto frontier for maximizing both x and y."""
    points = sorted(zip(x, y), reverse=True)
    pareto_x, pareto_y = [], []
    max_y = float('-inf')
    
    for px, py in points:
        if py > max_y:
            pareto_x.append(px)
            pareto_y.append(py)
            max_y = py
    
    return pareto_x, pareto_y


def create_defense_summary_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> str:
    """Create a LaTeX table summarizing defense results.
    
    Args:
        results: Dict mapping defense name to metrics dict
        metrics: List of metrics to include
        
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['asr_no_defense', 'asr_with_defense', 'reduction', 'clean_acc']
    
    defenses = list(results.keys())
    
    # Header
    header = "Defense & " + " & ".join(metrics) + " \\\\"
    
    # Rows
    rows = []
    for defense in defenses:
        values = [f"{results[defense].get(m, 0):.1f}" for m in metrics]
        rows.append(f"{defense} & " + " & ".join(values) + " \\\\")
    
    table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Defense Effectiveness Summary}}
\\begin{{tabular}}{{l{'c' * len(metrics)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return table
