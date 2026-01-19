"""Attention visualization utilities.

Provides visualization for attention patterns, attention ratios,
and layer-wise analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import torch


def plot_attention_ratio_distribution(
    attention_ratios: List[float],
    title: str = "Attention Ratio Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot distribution of attention ratios.
    
    Args:
        attention_ratios: List of attention ratio values (visual/text)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(attention_ratios, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle='--', label='Equal attention (ratio=1)')
    ax.axvline(x=np.mean(attention_ratios), color='green', linestyle='-', 
               label=f'Mean: {np.mean(attention_ratios):.2f}')
    
    ax.set_xlabel('Attention Ratio (Visual/Text)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_layerwise_hidden_norms(
    visual_norms: List[float],
    text_norms: List[float],
    layer_names: Optional[List[str]] = None,
    title: str = "Layer-wise Hidden State Norms",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot layer-wise hidden state norms for visual and text modalities.
    
    Args:
        visual_norms: Norms for visual tokens per layer
        text_norms: Norms for text tokens per layer
        layer_names: Optional layer names
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_layers = len(visual_norms)
    x = np.arange(num_layers)
    width = 0.35
    
    if layer_names is None:
        layer_names = [f'Layer {i}' for i in range(num_layers)]
    
    bars1 = ax.bar(x - width/2, visual_norms, width, label='Visual', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, text_norms, width, label='Text', color='orange', alpha=0.7)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Hidden State Norm')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis'
) -> plt.Figure:
    """Plot attention matrix as heatmap.
    
    Args:
        attention_matrix: 2D attention matrix
        x_labels: Labels for x-axis (keys)
        y_labels: Labels for y-axis (queries)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attention_matrix, cmap=cmap, aspect='auto')
    
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_visual_dominant_heads(
    head_ratios: Dict[Tuple[int, int], float],
    threshold: float = 3.0,
    title: str = "Visual-Dominant Attention Heads",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot visual-dominant attention heads.
    
    Args:
        head_ratios: Dict mapping (layer, head) to attention ratio
        threshold: Threshold for visual dominance
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract layers and heads
    layers = sorted(set(k[0] for k in head_ratios.keys()))
    heads = sorted(set(k[1] for k in head_ratios.keys()))
    
    # Create matrix
    matrix = np.zeros((len(layers), len(heads)))
    for (layer, head), ratio in head_ratios.items():
        layer_idx = layers.index(layer)
        head_idx = heads.index(head)
        matrix[layer_idx, head_idx] = ratio
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=max(5, threshold * 1.5))
    
    # Mark visual-dominant heads
    for (layer, head), ratio in head_ratios.items():
        if ratio > threshold:
            layer_idx = layers.index(layer)
            head_idx = heads.index(head)
            ax.scatter(head_idx, layer_idx, marker='*', s=100, c='black', zorder=5)
    
    ax.set_xticks(np.arange(len(heads)))
    ax.set_xticklabels([f'H{h}' for h in heads])
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels([f'L{l}' for l in layers])
    
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Layer')
    ax.set_title(f"{title}\n(★ = ratio > {threshold})")
    
    plt.colorbar(im, ax=ax, label='Attention Ratio (Visual/Text)')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attention_comparison(
    before_ratios: List[float],
    after_ratios: List[float],
    labels: List[str] = None,
    title: str = "Attention Ratio Before/After Rebalancing",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot comparison of attention ratios before and after intervention.
    
    Args:
        before_ratios: Attention ratios before intervention
        after_ratios: Attention ratios after intervention
        labels: Sample labels
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(before_ratios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before_ratios, width, label='Before', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, after_ratios, width, label='After', color='green', alpha=0.7)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal attention')
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Attention Ratio (Visual/Text)')
    ax.set_title(title)
    
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_attention_summary_figure(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """Create a summary figure with multiple attention visualizations.
    
    Args:
        results: Dictionary containing attention analysis results
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create 2x2 subplot grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Attention ratio distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if 'attention_ratios' in results:
        ratios = results['attention_ratios']
        ax1.hist(ratios, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=1.0, color='red', linestyle='--')
        ax1.axvline(x=np.mean(ratios), color='green', linestyle='-')
        ax1.set_xlabel('Attention Ratio')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Attention Ratio Distribution')
    
    # 2. Layer-wise norms
    ax2 = fig.add_subplot(gs[0, 1])
    if 'visual_norms' in results and 'text_norms' in results:
        num_layers = len(results['visual_norms'])
        x = np.arange(num_layers)
        ax2.plot(x, results['visual_norms'], 'b-o', label='Visual')
        ax2.plot(x, results['text_norms'], 'r-s', label='Text')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Hidden State Norm')
        ax2.set_title('Layer-wise Hidden State Norms')
        ax2.legend()
    
    # 3. Model comparison
    ax3 = fig.add_subplot(gs[1, 0])
    if 'model_ratios' in results:
        models = list(results['model_ratios'].keys())
        ratios = [results['model_ratios'][m] for m in models]
        ax3.bar(models, ratios, color=['blue', 'green', 'orange', 'red'][:len(models)])
        ax3.axhline(y=1.0, color='black', linestyle='--')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Mean Attention Ratio')
        ax3.set_title('Attention Ratio by Model')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Attack success vs attention ratio
    ax4 = fig.add_subplot(gs[1, 1])
    if 'asr_by_ratio' in results:
        ratio_bins = list(results['asr_by_ratio'].keys())
        asr_values = [results['asr_by_ratio'][b] for b in ratio_bins]
        ax4.bar(ratio_bins, asr_values, color='purple', alpha=0.7)
        ax4.set_xlabel('Attention Ratio Bin')
        ax4.set_ylabel('Attack Success Rate')
        ax4.set_title('ASR vs Attention Ratio')
    
    fig.suptitle('Attention Analysis Summary', fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
