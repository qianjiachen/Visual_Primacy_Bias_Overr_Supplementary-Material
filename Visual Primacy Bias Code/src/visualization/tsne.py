"""t-SNE visualization for modality gap analysis.

Visualizes the embedding space to show the gap between visual and text modalities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch


def compute_tsne_embeddings(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """Compute t-SNE embeddings.
    
    Args:
        embeddings: Input embeddings [N, D]
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
        
    Returns:
        2D t-SNE embeddings [N, 2]
    """
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        n_iter=n_iter,
        random_state=random_state
    )
    return tsne.fit_transform(embeddings)


def plot_modality_gap(
    visual_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Modality Gap Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    use_pca_init: bool = True
) -> plt.Figure:
    """Plot t-SNE visualization of modality gap.
    
    Args:
        visual_embeddings: Visual token embeddings [N_v, D]
        text_embeddings: Text token embeddings [N_t, D]
        labels: Optional labels for samples
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        use_pca_init: Whether to use PCA initialization for t-SNE
        
    Returns:
        Matplotlib figure
    """
    # Combine embeddings
    combined = np.vstack([visual_embeddings, text_embeddings])
    modality_labels = ['Visual'] * len(visual_embeddings) + ['Text'] * len(text_embeddings)
    
    # Apply PCA first if embeddings are high-dimensional
    if combined.shape[1] > 50:
        pca = PCA(n_components=50)
        combined = pca.fit_transform(combined)
    
    # Compute t-SNE
    tsne_embeddings = compute_tsne_embeddings(combined)
    
    # Split back
    visual_tsne = tsne_embeddings[:len(visual_embeddings)]
    text_tsne = tsne_embeddings[len(visual_embeddings):]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(visual_tsne[:, 0], visual_tsne[:, 1], 
               c='blue', alpha=0.6, label='Visual', s=50)
    ax.scatter(text_tsne[:, 0], text_tsne[:, 1], 
               c='red', alpha=0.6, label='Text', s=50)
    
    # Draw centroids
    visual_centroid = visual_tsne.mean(axis=0)
    text_centroid = text_tsne.mean(axis=0)
    
    ax.scatter(*visual_centroid, c='blue', marker='*', s=200, edgecolors='black', linewidths=2)
    ax.scatter(*text_centroid, c='red', marker='*', s=200, edgecolors='black', linewidths=2)
    
    # Draw line between centroids
    ax.plot([visual_centroid[0], text_centroid[0]], 
            [visual_centroid[1], text_centroid[1]], 
            'k--', alpha=0.5, linewidth=2)
    
    # Calculate and display gap
    gap = np.linalg.norm(visual_centroid - text_centroid)
    ax.annotate(f'Gap: {gap:.2f}', 
                xy=((visual_centroid[0] + text_centroid[0])/2, 
                    (visual_centroid[1] + text_centroid[1])/2),
                fontsize=12, ha='center')
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_attack_effect_tsne(
    clean_visual: np.ndarray,
    clean_text: np.ndarray,
    attack_visual: np.ndarray,
    attack_text: np.ndarray,
    title: str = "Attack Effect on Modality Gap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """Plot t-SNE comparison of clean vs attacked embeddings.
    
    Args:
        clean_visual: Clean visual embeddings
        clean_text: Clean text embeddings
        attack_visual: Attacked visual embeddings
        attack_text: Attacked text embeddings
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Clean embeddings
    combined_clean = np.vstack([clean_visual, clean_text])
    if combined_clean.shape[1] > 50:
        pca = PCA(n_components=50)
        combined_clean = pca.fit_transform(combined_clean)
    
    tsne_clean = compute_tsne_embeddings(combined_clean)
    
    axes[0].scatter(tsne_clean[:len(clean_visual), 0], 
                    tsne_clean[:len(clean_visual), 1],
                    c='blue', alpha=0.6, label='Visual', s=50)
    axes[0].scatter(tsne_clean[len(clean_visual):, 0], 
                    tsne_clean[len(clean_visual):, 1],
                    c='red', alpha=0.6, label='Text', s=50)
    axes[0].set_title('Clean Input')
    axes[0].legend()
    
    # Attacked embeddings
    combined_attack = np.vstack([attack_visual, attack_text])
    if combined_attack.shape[1] > 50:
        pca = PCA(n_components=50)
        combined_attack = pca.fit_transform(combined_attack)
    
    tsne_attack = compute_tsne_embeddings(combined_attack)
    
    axes[1].scatter(tsne_attack[:len(attack_visual), 0], 
                    tsne_attack[:len(attack_visual), 1],
                    c='blue', alpha=0.6, label='Visual', s=50)
    axes[1].scatter(tsne_attack[len(attack_visual):, 0], 
                    tsne_attack[len(attack_visual):, 1],
                    c='red', alpha=0.6, label='Text', s=50)
    axes[1].set_title('After TAS Attack')
    axes[1].legend()
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_embedding_trajectory(
    embeddings_over_layers: List[np.ndarray],
    modality: str = "visual",
    title: str = "Embedding Trajectory Across Layers",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot how embeddings evolve across layers.
    
    Args:
        embeddings_over_layers: List of embeddings at each layer
        modality: 'visual' or 'text'
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Stack all layer embeddings
    all_embeddings = np.vstack(embeddings_over_layers)
    layer_labels = []
    for i, emb in enumerate(embeddings_over_layers):
        layer_labels.extend([i] * len(emb))
    
    # Apply PCA then t-SNE
    if all_embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        all_embeddings = pca.fit_transform(all_embeddings)
    
    tsne_embeddings = compute_tsne_embeddings(all_embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by layer
    scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1],
                        c=layer_labels, cmap='viridis', alpha=0.6, s=30)
    
    # Draw trajectory through centroids
    centroids = []
    for i in range(len(embeddings_over_layers)):
        mask = np.array(layer_labels) == i
        centroid = tsne_embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    ax.plot(centroids[:, 0], centroids[:, 1], 'k-', linewidth=2, alpha=0.7)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, zorder=5)
    
    # Label start and end
    ax.annotate('Layer 0', centroids[0], fontsize=10, ha='right')
    ax.annotate(f'Layer {len(centroids)-1}', centroids[-1], fontsize=10, ha='left')
    
    plt.colorbar(scatter, ax=ax, label='Layer')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f"{title} ({modality.capitalize()} Tokens)")
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
