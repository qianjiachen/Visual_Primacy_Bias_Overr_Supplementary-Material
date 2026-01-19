"""Attention analysis for Visual Primacy Bias.

Implements attention ratio computation, layer-wise analysis,
visual-dominant head identification, and causal intervention.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

from src.models.interface import MLLMInterface


@dataclass
class AttentionAnalysisResult:
    """Result of attention analysis."""
    
    # Per-layer attention ratios
    layer_ratios: Dict[int, float]
    
    # Per-head attention ratios
    head_ratios: Dict[Tuple[int, int], float]
    
    # Visual-dominant heads (ratio > threshold)
    visual_dominant_heads: List[Tuple[int, int]]
    
    # Average attention ratio
    mean_ratio: float
    
    # Standard deviation
    std_ratio: float


class AttentionAnalyzer:
    """Analyzer for attention patterns in MLLMs.
    
    Computes normalized attention ratios between visual and system tokens
    to quantify Visual Primacy Bias.
    """
    
    def __init__(
        self,
        model: MLLMInterface,
        visual_dominant_threshold: float = 3.0,
    ):
        """Initialize attention analyzer.
        
        Args:
            model: MLLM interface
            visual_dominant_threshold: Threshold for visual-dominant head classification
        """
        self.model = model
        self.threshold = visual_dominant_threshold
    
    def compute_attention_ratio(
        self,
        attention_weights: torch.Tensor,
        visual_indices: List[int],
        system_indices: List[int],
        query_position: Optional[int] = None,
    ) -> float:
        """Compute normalized attention ratio.
        
        alpha_ratio = (1/N_v * sum_{j in V} A_ij) / (1/N_s * sum_{k in S} A_ik)
        
        Args:
            attention_weights: Attention matrix (seq_len, seq_len) or (num_heads, seq_len, seq_len)
            visual_indices: Indices of visual tokens
            system_indices: Indices of system prompt tokens
            query_position: Query position to analyze (default: last position)
        
        Returns:
            Normalized attention ratio
        """
        if len(visual_indices) == 0 or len(system_indices) == 0:
            logger.warning("Empty visual or system indices")
            return 1.0
        
        N_v = len(visual_indices)
        N_s = len(system_indices)
        
        # Handle different tensor shapes
        if attention_weights.dim() == 2:
            # Single head: (seq_len, seq_len)
            attn = attention_weights
        elif attention_weights.dim() == 3:
            # Multiple heads: (num_heads, seq_len, seq_len)
            # Average over heads
            attn = attention_weights.mean(dim=0)
        else:
            raise ValueError(f"Unexpected attention shape: {attention_weights.shape}")
        
        # Select query position
        if query_position is None:
            query_position = attn.shape[0] - 1  # Last position
        
        # Get attention row for query position
        attn_row = attn[query_position]
        
        # Compute normalized attention to visual tokens
        visual_attn = attn_row[visual_indices].sum().item() / N_v
        
        # Compute normalized attention to system tokens
        system_attn = attn_row[system_indices].sum().item() / N_s
        
        # Avoid division by zero
        if system_attn < 1e-10:
            return float('inf')
        
        ratio = visual_attn / system_attn
        
        return ratio
    
    def compute_attention_ratio_per_head(
        self,
        attention_weights: torch.Tensor,
        visual_indices: List[int],
        system_indices: List[int],
        query_position: Optional[int] = None,
    ) -> Dict[int, float]:
        """Compute attention ratio for each head.
        
        Args:
            attention_weights: Attention matrix (num_heads, seq_len, seq_len)
            visual_indices: Indices of visual tokens
            system_indices: Indices of system prompt tokens
            query_position: Query position to analyze
        
        Returns:
            Dictionary mapping head index to attention ratio
        """
        if attention_weights.dim() != 3:
            raise ValueError("Expected 3D tensor (num_heads, seq_len, seq_len)")
        
        num_heads = attention_weights.shape[0]
        ratios = {}
        
        for h in range(num_heads):
            ratios[h] = self.compute_attention_ratio(
                attention_weights[h],
                visual_indices,
                system_indices,
                query_position,
            )
        
        return ratios
    
    def analyze_layerwise(
        self,
        hidden_states: List[torch.Tensor],
        visual_indices: List[int],
        system_indices: List[int],
    ) -> Dict[str, List[float]]:
        """Analyze hidden state norms across layers.
        
        Args:
            hidden_states: List of hidden state tensors per layer
            visual_indices: Indices of visual tokens
            system_indices: Indices of system prompt tokens
        
        Returns:
            Dictionary with 'visual_norms' and 'system_norms' lists
        """
        visual_norms = []
        system_norms = []
        
        for layer_idx, hidden in enumerate(hidden_states):
            # hidden shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            if hidden.dim() == 3:
                hidden = hidden[0]  # Remove batch dimension
            
            # Compute average norm for visual tokens
            if visual_indices:
                visual_hidden = hidden[visual_indices]
                visual_norm = torch.norm(visual_hidden, dim=-1).mean().item()
            else:
                visual_norm = 0.0
            
            # Compute average norm for system tokens
            if system_indices:
                system_hidden = hidden[system_indices]
                system_norm = torch.norm(system_hidden, dim=-1).mean().item()
            else:
                system_norm = 0.0
            
            visual_norms.append(visual_norm)
            system_norms.append(system_norm)
        
        return {
            "visual_norms": visual_norms,
            "system_norms": system_norms,
        }
    
    def identify_visual_dominant_heads(
        self,
        attention_weights: torch.Tensor,
        visual_indices: List[int],
        system_indices: List[int],
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        """Identify visual-dominant attention heads.
        
        A head is visual-dominant if its attention ratio > threshold.
        
        Args:
            attention_weights: Attention tensor (num_layers, num_heads, seq_len, seq_len)
            visual_indices: Indices of visual tokens
            system_indices: Indices of system prompt tokens
            threshold: Classification threshold (default: self.threshold)
        
        Returns:
            List of (layer, head) tuples for visual-dominant heads
        """
        threshold = threshold or self.threshold
        visual_dominant = []
        
        if attention_weights.dim() == 4:
            num_layers, num_heads = attention_weights.shape[:2]
            
            for layer in range(num_layers):
                for head in range(num_heads):
                    ratio = self.compute_attention_ratio(
                        attention_weights[layer, head],
                        visual_indices,
                        system_indices,
                    )
                    
                    if ratio > threshold:
                        visual_dominant.append((layer, head))
        
        elif attention_weights.dim() == 3:
            # Single layer
            num_heads = attention_weights.shape[0]
            
            for head in range(num_heads):
                ratio = self.compute_attention_ratio(
                    attention_weights[head],
                    visual_indices,
                    system_indices,
                )
                
                if ratio > threshold:
                    visual_dominant.append((0, head))
        
        return visual_dominant
    
    def attention_rebalancing(
        self,
        attention_weights: torch.Tensor,
        visual_indices: List[int],
        gamma: float,
    ) -> torch.Tensor:
        """Apply attention rebalancing intervention.
        
        Scales visual attention weights by gamma and renormalizes.
        
        A'_ij = gamma * A_ij if j in V else A_ij
        Then renormalize so sum_j A'_ij = 1
        
        Args:
            attention_weights: Attention matrix
            visual_indices: Indices of visual tokens
            gamma: Scaling factor for visual attention (< 1 reduces visual attention)
        
        Returns:
            Rebalanced attention weights
        """
        # Clone to avoid modifying original
        rebalanced = attention_weights.clone()
        
        # Create mask for visual tokens
        visual_mask = torch.zeros(rebalanced.shape[-1], device=rebalanced.device)
        visual_mask[visual_indices] = 1.0
        
        # Apply scaling based on tensor dimensions
        if rebalanced.dim() == 2:
            # (seq_len, seq_len)
            for i in range(rebalanced.shape[0]):
                # Scale visual attention
                rebalanced[i] = rebalanced[i] * (1 - visual_mask) + \
                               rebalanced[i] * visual_mask * gamma
                # Renormalize
                rebalanced[i] = rebalanced[i] / rebalanced[i].sum()
        
        elif rebalanced.dim() == 3:
            # (num_heads, seq_len, seq_len)
            for h in range(rebalanced.shape[0]):
                for i in range(rebalanced.shape[1]):
                    rebalanced[h, i] = rebalanced[h, i] * (1 - visual_mask) + \
                                       rebalanced[h, i] * visual_mask * gamma
                    rebalanced[h, i] = rebalanced[h, i] / rebalanced[h, i].sum()
        
        elif rebalanced.dim() == 4:
            # (num_layers, num_heads, seq_len, seq_len)
            for l in range(rebalanced.shape[0]):
                for h in range(rebalanced.shape[1]):
                    for i in range(rebalanced.shape[2]):
                        rebalanced[l, h, i] = rebalanced[l, h, i] * (1 - visual_mask) + \
                                              rebalanced[l, h, i] * visual_mask * gamma
                        rebalanced[l, h, i] = rebalanced[l, h, i] / rebalanced[l, h, i].sum()
        
        return rebalanced
    
    def full_analysis(
        self,
        visual_indices: List[int],
        system_indices: List[int],
    ) -> AttentionAnalysisResult:
        """Perform full attention analysis on last forward pass.
        
        Args:
            visual_indices: Indices of visual tokens
            system_indices: Indices of system prompt tokens
        
        Returns:
            AttentionAnalysisResult with all computed metrics
        """
        # Get attention weights from model
        try:
            attention_weights = self.model.get_attention_weights()
        except RuntimeError as e:
            logger.error(f"Failed to get attention weights: {e}")
            raise
        
        layer_ratios = {}
        head_ratios = {}
        all_ratios = []
        
        if attention_weights.dim() == 4:
            num_layers, num_heads = attention_weights.shape[:2]
            
            for layer in range(num_layers):
                # Layer-level ratio (average over heads)
                layer_ratio = self.compute_attention_ratio(
                    attention_weights[layer],  # (num_heads, seq_len, seq_len)
                    visual_indices,
                    system_indices,
                )
                layer_ratios[layer] = layer_ratio
                
                # Per-head ratios
                for head in range(num_heads):
                    ratio = self.compute_attention_ratio(
                        attention_weights[layer, head],
                        visual_indices,
                        system_indices,
                    )
                    head_ratios[(layer, head)] = ratio
                    all_ratios.append(ratio)
        
        # Identify visual-dominant heads
        visual_dominant = [
            (l, h) for (l, h), r in head_ratios.items()
            if r > self.threshold
        ]
        
        # Compute statistics
        if all_ratios:
            mean_ratio = np.mean(all_ratios)
            std_ratio = np.std(all_ratios)
        else:
            mean_ratio = 0.0
            std_ratio = 0.0
        
        return AttentionAnalysisResult(
            layer_ratios=layer_ratios,
            head_ratios=head_ratios,
            visual_dominant_heads=visual_dominant,
            mean_ratio=mean_ratio,
            std_ratio=std_ratio,
        )
    
    def compute_hidden_state_norms(
        self,
        hidden_states: torch.Tensor,
        token_indices: List[int],
    ) -> List[float]:
        """Compute hidden state norms for specific tokens across layers.
        
        Args:
            hidden_states: Hidden states (num_layers, seq_len, hidden_dim)
            token_indices: Indices of tokens to analyze
        
        Returns:
            List of average norms per layer
        """
        norms = []
        
        for layer_idx in range(hidden_states.shape[0]):
            layer_hidden = hidden_states[layer_idx]
            
            if token_indices:
                token_hidden = layer_hidden[token_indices]
                avg_norm = torch.norm(token_hidden, dim=-1).mean().item()
            else:
                avg_norm = 0.0
            
            norms.append(avg_norm)
        
        return norms
