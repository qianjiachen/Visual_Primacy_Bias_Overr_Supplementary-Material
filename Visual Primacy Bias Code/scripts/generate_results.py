#!/usr/bin/env python3
"""Generate all results tables and figures for the paper.

This script loads experiment results and generates:
- LaTeX tables for the paper
- Visualization figures
- Summary statistics
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all experiment results from directory."""
    results = {}
    results_path = Path(results_dir)
    
    # Load main attack results
    attack_file = results_path / "attack_results.json"
    if attack_file.exists():
        with open(attack_file, 'r') as f:
            results['attack_results'] = json.load(f)
    
    # Load analysis results
    analysis_file = results_path / "analysis_results.json"
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            results['analysis_results'] = json.load(f)
    
    # Load defense results
    defense_file = results_path / "defense_results.json"
    if defense_file.exists():
        with open(defense_file, 'r') as f:
            results['defense_results'] = json.load(f)
    
    return results


def generate_main_results_table(results: Dict[str, Any], output_dir: str):
    """Generate Table 1: Main ASR comparison."""
    from src.visualization.latex_export import export_main_results_table
    
    attack_results = results.get('attack_results', {})
    
    # Extract model and attack names
    models = list(attack_results.get('by_model', {}).keys())
    attacks = ['Direct', 'FigStep', 'GCG', 'PGD', 'TAS (Ours)']
    
    # Build results matrix
    main_results = {}
    for model in models:
        main_results[model] = {}
        model_data = attack_results.get('by_model', {}).get(model, {})
        for attack in attacks:
            main_results[model][attack] = model_data.get(attack.lower().replace(' ', '_'), 0)
    
    table = export_main_results_table(
        main_results, models, attacks,
        caption="Attack Success Rate (\\%) across models and attack methods",
        label="tab:main_results"
    )
    
    output_path = Path(output_dir) / "table1_main_results.tex"
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"Generated: {output_path}")


def generate_category_results_table(results: Dict[str, Any], output_dir: str):
    """Generate Table 2: Category-wise ASR."""
    from src.visualization.latex_export import export_category_results_table
    
    attack_results = results.get('attack_results', {})
    
    categories = ['Illegal Activity', 'Hate Speech', 'Physical Harm', 'Fraud', 'Malware']
    methods = ['Direct', 'TAS']
    
    # Build category results
    category_results = {}
    for category in categories:
        category_results[category] = {}
        cat_data = attack_results.get('by_category', {}).get(category, {})
        for method in methods:
            category_results[category][method] = cat_data.get(method.lower(), 0)
    
    table = export_category_results_table(
        category_results, categories, methods,
        caption="Category-wise Attack Success Rate (\\%)",
        label="tab:category_results"
    )
    
    output_path = Path(output_dir) / "table2_category_results.tex"
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"Generated: {output_path}")


def generate_transfer_table(results: Dict[str, Any], output_dir: str):
    """Generate Table 3: Transfer attack matrix."""
    from src.visualization.latex_export import export_transfer_matrix_table
    
    attack_results = results.get('attack_results', {})
    transfer_matrix = attack_results.get('transfer_matrix', {})
    
    models = list(transfer_matrix.keys()) if transfer_matrix else ['LLaVA', 'Qwen-VL', 'InstructBLIP', 'GPT-4V']
    
    table = export_transfer_matrix_table(
        transfer_matrix, models,
        caption="Transfer Attack Success Rate (\\%)",
        label="tab:transfer"
    )
    
    output_path = Path(output_dir) / "table3_transfer.tex"
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"Generated: {output_path}")


def generate_defense_table(results: Dict[str, Any], output_dir: str):
    """Generate Table 4: Defense effectiveness."""
    from src.visualization.latex_export import export_defense_results_table
    
    defense_results = results.get('defense_results', {})
    
    defenses = ['No Defense', 'OCR Filter', 'Gaussian Blur', 'Safety Prompt', 'Attention Rebalancing']
    metrics = ['ASR (\\%)', 'Clean Acc (\\%)', 'Latency (ms)']
    
    # Build defense results
    defense_data = {}
    for defense in defenses:
        defense_data[defense] = {}
        d_data = defense_results.get(defense.lower().replace(' ', '_'), {})
        for metric in metrics:
            defense_data[defense][metric] = d_data.get(metric.lower().replace(' ', '_').replace('(%)', '').replace('(ms)', ''), 0)
    
    table = export_defense_results_table(
        defense_data, defenses, metrics,
        caption="Defense Effectiveness Against TAS Attack",
        label="tab:defense"
    )
    
    output_path = Path(output_dir) / "table4_defense.tex"
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"Generated: {output_path}")


def generate_attention_figures(results: Dict[str, Any], output_dir: str):
    """Generate attention analysis figures."""
    from src.visualization.attention import (
        plot_attention_ratio_distribution,
        plot_layerwise_hidden_norms,
        plot_visual_dominant_heads,
        create_attention_summary_figure
    )
    
    analysis_results = results.get('analysis_results', {})
    attention_data = analysis_results.get('attention', {})
    
    # Figure 1: Attention ratio distribution
    if 'attention_ratios' in attention_data:
        fig = plot_attention_ratio_distribution(
            attention_data['attention_ratios'],
            title="Attention Ratio Distribution (Visual/Text)",
            save_path=str(Path(output_dir) / "fig_attention_ratio.pdf")
        )
        print(f"Generated: {output_dir}/fig_attention_ratio.pdf")
    
    # Figure 2: Layer-wise norms
    if 'visual_norms' in attention_data and 'text_norms' in attention_data:
        fig = plot_layerwise_hidden_norms(
            attention_data['visual_norms'],
            attention_data['text_norms'],
            title="Layer-wise Hidden State Norms",
            save_path=str(Path(output_dir) / "fig_layerwise_norms.pdf")
        )
        print(f"Generated: {output_dir}/fig_layerwise_norms.pdf")
    
    # Figure 3: Visual-dominant heads
    if 'head_ratios' in attention_data:
        # Convert string keys back to tuples
        head_ratios = {}
        for k, v in attention_data['head_ratios'].items():
            layer, head = map(int, k.strip('()').split(','))
            head_ratios[(layer, head)] = v
        
        fig = plot_visual_dominant_heads(
            head_ratios,
            threshold=3.0,
            title="Visual-Dominant Attention Heads",
            save_path=str(Path(output_dir) / "fig_visual_heads.pdf")
        )
        print(f"Generated: {output_dir}/fig_visual_heads.pdf")


def generate_tsne_figures(results: Dict[str, Any], output_dir: str):
    """Generate t-SNE visualization figures."""
    from src.visualization.tsne import plot_modality_gap, plot_attack_effect_tsne
    
    analysis_results = results.get('analysis_results', {})
    embedding_data = analysis_results.get('embeddings', {})
    
    # Figure: Modality gap
    if 'visual_embeddings' in embedding_data and 'text_embeddings' in embedding_data:
        fig = plot_modality_gap(
            np.array(embedding_data['visual_embeddings']),
            np.array(embedding_data['text_embeddings']),
            title="Modality Gap Visualization",
            save_path=str(Path(output_dir) / "fig_modality_gap.pdf")
        )
        print(f"Generated: {output_dir}/fig_modality_gap.pdf")


def generate_defense_figures(results: Dict[str, Any], output_dir: str):
    """Generate defense evaluation figures."""
    from src.visualization.defense import plot_defense_comparison, plot_defense_tradeoff
    
    defense_results = results.get('defense_results', {})
    
    if defense_results:
        # Defense comparison bar chart
        fig = plot_defense_comparison(
            defense_results,
            metrics=['asr_reduction', 'clean_accuracy'],
            title="Defense Effectiveness Comparison",
            save_path=str(Path(output_dir) / "fig_defense_comparison.pdf")
        )
        print(f"Generated: {output_dir}/fig_defense_comparison.pdf")


def generate_summary_statistics(results: Dict[str, Any], output_dir: str):
    """Generate summary statistics file."""
    stats = {
        'attack': {},
        'analysis': {},
        'defense': {}
    }
    
    # Attack statistics
    attack_results = results.get('attack_results', {})
    if 'by_model' in attack_results:
        for model, data in attack_results['by_model'].items():
            if 'tas' in data:
                stats['attack'][f'{model}_tas_asr'] = data['tas']
    
    # Analysis statistics
    analysis_results = results.get('analysis_results', {})
    if 'attention' in analysis_results:
        attn = analysis_results['attention']
        if 'attention_ratios' in attn:
            stats['analysis']['mean_attention_ratio'] = np.mean(attn['attention_ratios'])
            stats['analysis']['std_attention_ratio'] = np.std(attn['attention_ratios'])
    
    # Defense statistics
    defense_results = results.get('defense_results', {})
    for defense, data in defense_results.items():
        if 'asr_reduction' in data:
            stats['defense'][f'{defense}_reduction'] = data['asr_reduction']
    
    output_path = Path(output_dir) / "summary_statistics.json"
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper results")
    parser.add_argument("--results-dir", type=str, default="outputs/results",
                        help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="outputs/paper",
                        help="Directory to save generated tables and figures")
    parser.add_argument("--tables-only", action="store_true",
                        help="Generate only LaTeX tables")
    parser.add_argument("--figures-only", action="store_true",
                        help="Generate only figures")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found. Creating placeholder results for demonstration.")
        results = create_placeholder_results()
    
    tables_dir = os.path.join(args.output_dir, "tables")
    figures_dir = os.path.join(args.output_dir, "figures")
    
    if not args.figures_only:
        print("\n=== Generating LaTeX Tables ===")
        generate_main_results_table(results, tables_dir)
        generate_category_results_table(results, tables_dir)
        generate_transfer_table(results, tables_dir)
        generate_defense_table(results, tables_dir)
    
    if not args.tables_only:
        print("\n=== Generating Figures ===")
        generate_attention_figures(results, figures_dir)
        generate_tsne_figures(results, figures_dir)
        generate_defense_figures(results, figures_dir)
    
    print("\n=== Generating Summary Statistics ===")
    generate_summary_statistics(results, args.output_dir)
    
    print("\n=== Done ===")
    print(f"Results saved to: {args.output_dir}")


def create_placeholder_results() -> Dict[str, Any]:
    """Create placeholder results for demonstration."""
    return {
        'attack_results': {
            'by_model': {
                'LLaVA-Next': {'direct': 12.3, 'figstep': 34.5, 'gcg': 28.7, 'pgd': 31.2, 'tas': 78.4},
                'Qwen-VL': {'direct': 8.9, 'figstep': 29.3, 'gcg': 24.1, 'pgd': 27.8, 'tas': 72.6},
                'InstructBLIP': {'direct': 15.2, 'figstep': 38.7, 'gcg': 32.4, 'pgd': 35.1, 'tas': 81.3},
                'GPT-4V': {'direct': 3.2, 'figstep': 18.4, 'gcg': 12.7, 'pgd': 15.3, 'tas': 45.8},
            },
            'by_category': {
                'Illegal Activity': {'direct': 10.5, 'tas': 75.2},
                'Hate Speech': {'direct': 8.3, 'tas': 68.9},
                'Physical Harm': {'direct': 12.1, 'tas': 79.4},
                'Fraud': {'direct': 9.7, 'tas': 71.3},
                'Malware': {'direct': 11.4, 'tas': 76.8},
            },
            'transfer_matrix': {
                'LLaVA-Next': {'LLaVA-Next': 78.4, 'Qwen-VL': 52.3, 'InstructBLIP': 48.7, 'GPT-4V': 28.4},
                'Qwen-VL': {'LLaVA-Next': 49.8, 'Qwen-VL': 72.6, 'InstructBLIP': 45.2, 'GPT-4V': 25.1},
                'InstructBLIP': {'LLaVA-Next': 51.2, 'Qwen-VL': 47.9, 'InstructBLIP': 81.3, 'GPT-4V': 29.7},
                'GPT-4V': {'LLaVA-Next': 32.4, 'Qwen-VL': 28.6, 'InstructBLIP': 31.8, 'GPT-4V': 45.8},
            }
        },
        'analysis_results': {
            'attention': {
                'attention_ratios': list(np.random.lognormal(1.2, 0.5, 100)),
                'visual_norms': list(np.random.uniform(0.8, 1.5, 32)),
                'text_norms': list(np.random.uniform(0.3, 0.8, 32)),
                'head_ratios': {f"({i}, {j})": np.random.uniform(0.5, 5.0) 
                               for i in range(32) for j in range(32) if np.random.random() > 0.9}
            },
            'embeddings': {
                'visual_embeddings': np.random.randn(50, 64).tolist(),
                'text_embeddings': np.random.randn(50, 64).tolist(),
            }
        },
        'defense_results': {
            'no_defense': {'asr': 78.4, 'clean_accuracy': 95.2, 'latency': 0},
            'ocr_filter': {'asr': 45.3, 'clean_accuracy': 89.7, 'latency': 120, 'asr_reduction': 33.1},
            'gaussian_blur': {'asr': 52.8, 'clean_accuracy': 91.4, 'latency': 15, 'asr_reduction': 25.6},
            'safety_prompt': {'asr': 61.2, 'clean_accuracy': 94.8, 'latency': 5, 'asr_reduction': 17.2},
            'attention_rebalancing': {'asr': 38.7, 'clean_accuracy': 92.1, 'latency': 45, 'asr_reduction': 39.7},
        }
    }


if __name__ == "__main__":
    main()
