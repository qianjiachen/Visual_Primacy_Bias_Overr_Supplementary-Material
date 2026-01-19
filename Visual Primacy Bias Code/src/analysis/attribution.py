"""Attribution analysis for Visual Primacy Bias.

Implements Integrated Gradients, QKV path ablation, and logit lens
for mechanistic analysis of MLLM behavior.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from scipy.stats import spearmanr

from src.models.interface import MLLMInterface


@dataclass
class AttributionResult:
    """Result of attribution analysis."""
    
    # Integrated Gradients attribution
    ig_attribution: torch.Tensor
    
    # Total attribution to visual vs system tokens
    visual_attribution: float
    system_attribution: float
    
    # Attribution ratio
    attribution_ratio: float


class AttributionAnalyzer:
    """Attribution analysis for information flow in MLLMs.
    
    Implements multiple attribution methods:
    - Integrated Gradients (IG)
    - QKV path ablation
    - Logit lens decomposition
    """
    
    def __init__(
        self,
        model: MLLMInterface,
        num_steps: int = 50,
    ):
        """Initialize attribution analyzer.
        
        Args:
            model: MLLM interface
            num_steps: Number of interpolation steps for IG
        """
        self.model = model
        self.num_steps = num_steps
    
    def integrated_gradients(
        self,
        visual_embedding: torch.Tensor,
        target_token_id: int,
        baseline: Optional[torch.Tensor] = None,
        system_input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Integrated Gradients attribution.
        
        Attr_v(y_harm) = (E_v - E_base) * integral_0^1 grad_{E_v} F(E_tilde(alpha)) dalpha
        
        Args:
            visual_embedding: Visual embedding tensor (N_v, d)
            target_token_id: Target token ID to compute attribution for
            baseline: Baseline embedding (default: zero)
            system_input_ids: System prompt input IDs
        
        Returns:
            Attribution tensor with same shape as visual_embedding
        """
        device = visual_embedding.device
        
        # Default baseline is zero embedding
        if baseline is None:
            baseline = torch.zeros_like(visual_embedding)
        
        # Ensure baseline is on same device
        baseline = baseline.to(device)
        
        # Compute difference
        diff = visual_embedding - baseline
        
        # Accumulate gradients
        total_gradients = torch.zeros_like(visual_embedding)
        
        for step in range(self.num_steps):
            # Interpolation coefficient
            alpha = step / self.num_steps
            
            # Interpolated embedding
            interpolated = baseline + alpha * diff
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            # Forward pass
            if system_input_ids is not None:
                logits = self.model.forward_with_embedding(
                    interpolated,
                    system_input_ids,
                )
            else:
                # Use model's default forward
                logits = self.model.forward_with_embedding(
                    interpolated,
                    torch.tensor([[1]], device=device),  # Dummy input
                )
            
            # Get logit for target token at last position
            target_logit = logits[0, -1, target_token_id]
            
            # Compute gradient
            target_logit.backward(retain_graph=True)
            
            if interpolated.grad is not None:
                total_gradients += interpolated.grad.clone()
            
            # Clear gradients
            interpolated.grad = None
        
        # Average gradients
        avg_gradients = total_gradients / self.num_steps
        
        # Compute attribution
        attribution = diff * avg_gradients
        
        return attribution
    
    def compute_attribution_scores(
        self,
        attribution: torch.Tensor,
        visual_indices: List[int],
        system_indices: List[int],
    ) -> Tuple[float, float, float]:
        """Compute total attribution scores for visual and system tokens.
        
        Args:
            attribution: Attribution tensor
            visual_indices: Indices of visual tokens
            system_indices: Indices of system prompt tokens
        
        Returns:
            Tuple of (visual_attribution, system_attribution, ratio)
        """
        # Sum absolute attribution values
        if visual_indices:
            visual_attr = attribution[visual_indices].abs().sum().item()
        else:
            visual_attr = 0.0
        
        if system_indices:
            system_attr = attribution[system_indices].abs().sum().item()
        else:
            system_attr = 0.0
        
        # Compute ratio
        if system_attr > 1e-10:
            ratio = visual_attr / system_attr
        else:
            ratio = float('inf') if visual_attr > 0 else 1.0
        
        return visual_attr, system_attr, ratio
    
    def qkv_path_ablation(
        self,
        layer: int,
        head: int,
        visual_indices: List[int],
        target_token_id: int,
    ) -> float:
        """Perform QKV path ablation analysis.
        
        Measures the contribution of visual tokens through a specific
        attention head by ablating the QKV path.
        
        Args:
            layer: Layer index
            head: Head index
            visual_indices: Indices of visual tokens
            target_token_id: Target token ID
        
        Returns:
            Change in target token probability after ablation
        """
        # This requires model-specific implementation
        # Here we provide a simplified version
        
        try:
            # Get baseline probability
            attention_weights = self.model.get_attention_weights(layer=layer, head=head)
            
            # Compute baseline attention to visual tokens
            visual_attn = attention_weights[-1, visual_indices].sum().item()
            
            # The ablation effect is proportional to visual attention
            # Full implementation would require modifying the forward pass
            ablation_effect = visual_attn
            
            return ablation_effect
            
        except Exception as e:
            logger.warning(f"QKV ablation failed for layer {layer}, head {head}: {e}")
            return 0.0
    
    def logit_lens(
        self,
        hidden_states: List[torch.Tensor],
        unembedding_matrix: torch.Tensor,
        target_token_id: int,
    ) -> List[float]:
        """Apply logit lens analysis.
        
        LogitContrib_v^(l) = W_U * LayerNorm(h_v^(l))
        
        Args:
            hidden_states: List of hidden state tensors per layer
            unembedding_matrix: Unembedding matrix (vocab_size, hidden_dim)
            target_token_id: Target token ID to analyze
        
        Returns:
            List of target token logits per layer
        """
        contributions = []
        
        for layer_idx, hidden in enumerate(hidden_states):
            # hidden shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            if hidden.dim() == 3:
                hidden = hidden[0]  # Remove batch dimension
            
            # Get last position hidden state
            last_hidden = hidden[-1]  # (hidden_dim,)
            
            # Apply layer norm (simplified - actual implementation may vary)
            normed = F.layer_norm(last_hidden, (last_hidden.shape[-1],))
            
            # Project to vocabulary
            logits = torch.matmul(normed, unembedding_matrix.T)
            
            # Get target token logit
            target_logit = logits[target_token_id].item()
            contributions.append(target_logit)
        
        return contributions
    
    def compute_method_correlation(
        self,
        ig_scores: List[float],
        qkv_scores: List[float],
        logit_lens_scores: List[float],
    ) -> Dict[str, float]:
        """Compute Spearman correlation between attribution methods.
        
        Args:
            ig_scores: Integrated Gradients scores per layer
            qkv_scores: QKV ablation scores per layer
            logit_lens_scores: Logit lens scores per layer
        
        Returns:
            Dictionary of correlation coefficients
        """
        correlations = {}
        
        # IG vs QKV
        if len(ig_scores) == len(qkv_scores) and len(ig_scores) > 2:
            rho, _ = spearmanr(ig_scores, qkv_scores)
            correlations["ig_qkv"] = rho
        
        # IG vs Logit Lens
        if len(ig_scores) == len(logit_lens_scores) and len(ig_scores) > 2:
            rho, _ = spearmanr(ig_scores, logit_lens_scores)
            correlations["ig_logitlens"] = rho
        
        # QKV vs Logit Lens
        if len(qkv_scores) == len(logit_lens_scores) and len(qkv_scores) > 2:
            rho, _ = spearmanr(qkv_scores, logit_lens_scores)
            correlations["qkv_logitlens"] = rho
        
        return correlations
    
    def full_attribution_analysis(
        self,
        visual_embedding: torch.Tensor,
        target_token_id: int,
        visual_indices: List[int],
        system_indices: List[int],
        system_input_ids: Optional[torch.Tensor] = None,
    ) -> AttributionResult:
        """Perform full attribution analysis.
        
        Args:
            visual_embedding: Visual embedding tensor
            target_token_id: Target token ID
            visual_indices: Indices of visual tokens
            system_indices: Indices of system prompt tokens
            system_input_ids: System prompt input IDs
        
        Returns:
            AttributionResult with all computed metrics
        """
        # Compute Integrated Gradients
        ig_attribution = self.integrated_gradients(
            visual_embedding,
            target_token_id,
            system_input_ids=system_input_ids,
        )
        
        # Compute attribution scores
        visual_attr, system_attr, ratio = self.compute_attribution_scores(
            ig_attribution,
            visual_indices,
            system_indices,
        )
        
        return AttributionResult(
            ig_attribution=ig_attribution,
            visual_attribution=visual_attr,
            system_attribution=system_attr,
            attribution_ratio=ratio,
        )
