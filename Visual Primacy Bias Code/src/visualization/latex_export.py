"""LaTeX table export utilities.

Generates LaTeX tables for paper results.
"""

from typing import Dict, List, Optional, Any
import numpy as np


def export_main_results_table(
    results: Dict[str, Dict[str, float]],
    models: List[str],
    attacks: List[str],
    metric: str = "asr",
    caption: str = "Attack Success Rate (\\%) Comparison",
    label: str = "tab:main_results"
) -> str:
    """Export main results table in LaTeX format.
    
    Args:
        results: Nested dict [model][attack] -> value
        models: List of model names
        attacks: List of attack names
        metric: Metric name for caption
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    # Header
    header = "Model & " + " & ".join(attacks) + " \\\\"
    
    # Rows
    rows = []
    for model in models:
        values = []
        for attack in attacks:
            val = results.get(model, {}).get(attack, 0)
            # Bold best value per row
            values.append(f"{val:.1f}")
        rows.append(f"{model} & " + " & ".join(values) + " \\\\")
    
    table = f"""\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * len(attacks)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""
    
    return table


def export_category_results_table(
    results: Dict[str, Dict[str, float]],
    categories: List[str],
    methods: List[str],
    caption: str = "Category-wise Attack Success Rate (\\%)",
    label: str = "tab:category_results"
) -> str:
    """Export category-wise results table.
    
    Args:
        results: Nested dict [category][method] -> value
        categories: List of harm categories
        methods: List of attack methods
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    header = "Category & " + " & ".join(methods) + " \\\\"
    
    rows = []
    for category in categories:
        values = []
        for method in methods:
            val = results.get(category, {}).get(method, 0)
            values.append(f"{val:.1f}")
        rows.append(f"{category} & " + " & ".join(values) + " \\\\")
    
    # Add average row
    avg_values = []
    for method in methods:
        vals = [results.get(cat, {}).get(method, 0) for cat in categories]
        avg_values.append(f"{np.mean(vals):.1f}")
    rows.append("\\midrule")
    rows.append(f"\\textbf{{Average}} & " + " & ".join(avg_values) + " \\\\")
    
    table = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * len(methods)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return table


def export_transfer_matrix_table(
    transfer_matrix: Dict[str, Dict[str, float]],
    models: List[str],
    caption: str = "Transfer Attack Success Rate (\\%)",
    label: str = "tab:transfer"
) -> str:
    """Export transfer attack matrix table.
    
    Args:
        transfer_matrix: Nested dict [source_model][target_model] -> ASR
        models: List of model names
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    header = "Source $\\rightarrow$ Target & " + " & ".join(models) + " \\\\"
    
    rows = []
    for source in models:
        values = []
        for target in models:
            val = transfer_matrix.get(source, {}).get(target, 0)
            if source == target:
                values.append(f"\\textbf{{{val:.1f}}}")  # Bold diagonal
            else:
                values.append(f"{val:.1f}")
        rows.append(f"{source} & " + " & ".join(values) + " \\\\")
    
    table = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * len(models)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return table


def export_defense_results_table(
    results: Dict[str, Dict[str, float]],
    defenses: List[str],
    metrics: List[str] = None,
    caption: str = "Defense Effectiveness",
    label: str = "tab:defense"
) -> str:
    """Export defense evaluation results table.
    
    Args:
        results: Nested dict [defense][metric] -> value
        defenses: List of defense names
        metrics: List of metrics to include
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ['ASR (\\%)', 'Clean Acc (\\%)', 'Latency (ms)']
    
    header = "Defense & " + " & ".join(metrics) + " \\\\"
    
    rows = []
    for defense in defenses:
        values = []
        for metric in metrics:
            val = results.get(defense, {}).get(metric, 0)
            values.append(f"{val:.1f}")
        rows.append(f"{defense} & " + " & ".join(values) + " \\\\")
    
    table = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * len(metrics)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return table


def export_attention_analysis_table(
    results: Dict[str, Dict[str, float]],
    models: List[str],
    caption: str = "Attention Analysis Results",
    label: str = "tab:attention"
) -> str:
    """Export attention analysis results table.
    
    Args:
        results: Nested dict [model][metric] -> value
        models: List of model names
        caption: Table caption
        label: Table label
        
    Returns:
        LaTeX table string
    """
    metrics = ['Attn Ratio', 'Visual Heads', 'Text Heads', 'Rebalance Effect']
    header = "Model & " + " & ".join(metrics) + " \\\\"
    
    rows = []
    for model in models:
        values = []
        for metric in metrics:
            val = results.get(model, {}).get(metric, 0)
            if isinstance(val, float):
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))
        rows.append(f"{model} & " + " & ".join(values) + " \\\\")
    
    table = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{l{'c' * len(metrics)}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    return table


def export_all_tables(
    experiment_results: Dict[str, Any],
    output_dir: str = "outputs/tables"
) -> Dict[str, str]:
    """Export all result tables to LaTeX files.
    
    Args:
        experiment_results: Complete experiment results dictionary
        output_dir: Directory to save tables
        
    Returns:
        Dict mapping table name to file path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    tables = {}
    
    # Main results
    if 'main_results' in experiment_results:
        table = export_main_results_table(
            experiment_results['main_results'],
            experiment_results.get('models', []),
            experiment_results.get('attacks', [])
        )
        path = os.path.join(output_dir, 'main_results.tex')
        with open(path, 'w') as f:
            f.write(table)
        tables['main_results'] = path
    
    # Category results
    if 'category_results' in experiment_results:
        table = export_category_results_table(
            experiment_results['category_results'],
            experiment_results.get('categories', []),
            experiment_results.get('methods', [])
        )
        path = os.path.join(output_dir, 'category_results.tex')
        with open(path, 'w') as f:
            f.write(table)
        tables['category_results'] = path
    
    # Transfer matrix
    if 'transfer_matrix' in experiment_results:
        table = export_transfer_matrix_table(
            experiment_results['transfer_matrix'],
            experiment_results.get('models', [])
        )
        path = os.path.join(output_dir, 'transfer_matrix.tex')
        with open(path, 'w') as f:
            f.write(table)
        tables['transfer_matrix'] = path
    
    # Defense results
    if 'defense_results' in experiment_results:
        table = export_defense_results_table(
            experiment_results['defense_results'],
            experiment_results.get('defenses', [])
        )
        path = os.path.join(output_dir, 'defense_results.tex')
        with open(path, 'w') as f:
            f.write(table)
        tables['defense_results'] = path
    
    return tables
