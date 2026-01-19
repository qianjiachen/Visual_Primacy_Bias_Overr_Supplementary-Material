"""Visualization utilities."""

from .attention import (
    plot_attention_ratio_distribution,
    plot_layerwise_hidden_norms,
    plot_attention_heatmap,
    plot_visual_dominant_heads,
    plot_attention_comparison,
    create_attention_summary_figure,
)
from .tsne import (
    compute_tsne_embeddings,
    plot_modality_gap,
    plot_attack_effect_tsne,
    plot_embedding_trajectory,
)
from .defense import (
    plot_defense_comparison,
    plot_asr_by_defense_level,
    plot_defense_tradeoff,
    create_defense_summary_table,
)
from .latex_export import (
    export_main_results_table,
    export_category_results_table,
    export_transfer_matrix_table,
    export_defense_results_table,
    export_attention_analysis_table,
    export_all_tables,
)

__all__ = [
    # Attention visualization
    'plot_attention_ratio_distribution',
    'plot_layerwise_hidden_norms',
    'plot_attention_heatmap',
    'plot_visual_dominant_heads',
    'plot_attention_comparison',
    'create_attention_summary_figure',
    # t-SNE visualization
    'compute_tsne_embeddings',
    'plot_modality_gap',
    'plot_attack_effect_tsne',
    'plot_embedding_trajectory',
    # Defense visualization
    'plot_defense_comparison',
    'plot_asr_by_defense_level',
    'plot_defense_tradeoff',
    'create_defense_summary_table',
    # LaTeX export
    'export_main_results_table',
    'export_category_results_table',
    'export_transfer_matrix_table',
    'export_defense_results_table',
    'export_attention_analysis_table',
    'export_all_tables',
]

__all__ = []
