"""Gradient-guided style selection for TAS attack.

Implements the scoring function and candidate selection algorithm
for Typographic Attention Steering.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm

from src.models.interface import MLLMInterface
from src.models.config import DistortionParams


@dataclass
class ScoredCandidate:
    """A candidate image with its computed score."""
    image: np.ndarray
    font: str
    distortion: DistortionParams
    score: float
    gradient_alignment: float
    semantic_similarity: float
    visual_embedding: Optional[torch.Tensor] = None


class GradientScorer:
    """Gradient-guided scoring for TAS candidate selection.
    
    Implements the scoring function:
    Score(E_v, X_sys, Y_harm) = cos(g, E_v) + lambda * sim(E_v, E_harm)
    
    where:
    - g = gradient of loss w.r.t. visual embedding
    - E_v = visual embedding of candidate image
    - E_harm = embedding of harmful instruction text
    - lambda = weight balancing gradient alignment and semantic similarity
    """
    
    def __init__(
        self,
        model: MLLMInterface,
        lambda_weight: float = 0.3,
        device: str = "cuda",
    ):
        """Initialize the gradient scorer.
        
        Args:
            model: MLLM interface for computing embeddings and gradients
            lambda_weight: Weight for semantic similarity term (default 0.3)
            device: Device for computations
        """
        self.model = model
        self.lambda_weight = lambda_weight
        self.device = device
        
        # Cache for text embeddings
        self._harm_embedding_cache: Dict[str, torch.Tensor] = {}
    
    def compute_visual_embedding(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        """Compute visual embedding for an image.
        
        E_v = P(ViT(I))
        
        Args:
            image: Input image (H, W, 3) in RGB format
        
        Returns:
            Visual embedding tensor
        """
        embedding = self.model.encode_image(image)
        return embedding.to(self.device)
    
    def compute_harm_embedding(
        self,
        harm_text: str,
    ) -> torch.Tensor:
        """Compute embedding for harmful instruction text.
        
        Args:
            harm_text: Harmful instruction text
        
        Returns:
            Text embedding tensor
        """
        if harm_text not in self._harm_embedding_cache:
            embedding = self.model.encode_text(harm_text)
            # Average over sequence length to get single vector
            embedding = embedding.mean(dim=0)
            self._harm_embedding_cache[harm_text] = embedding.to(self.device)
        
        return self._harm_embedding_cache[harm_text]
    
    def compute_gradient(
        self,
        visual_embedding: torch.Tensor,
        system_prompt: str,
        target_response: str,
    ) -> torch.Tensor:
        """Compute gradient of target loss w.r.t. visual embedding.
        
        g = grad_{E_v} L(Y_harm | E_v, X_sys)
        
        Args:
            visual_embedding: Visual embedding tensor
            system_prompt: System prompt text
            target_response: Target harmful response
        
        Returns:
            Gradient tensor with same shape as visual_embedding
        """
        # Enable gradient computation
        visual_embedding = visual_embedding.clone().detach().requires_grad_(True)
        
        # Tokenize system prompt
        system_ids = self.model.processor.tokenizer.encode(
            system_prompt,
            return_tensors="pt",
        ).to(self.device)
        
        # Tokenize target response
        target_ids = self.model.processor.tokenizer.encode(
            target_response,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        
        # Forward pass with visual embedding
        logits = self.model.forward_with_embedding(
            visual_embedding,
            system_ids,
        )
        
        # Compute cross-entropy loss for target tokens
        # We want to maximize probability of target response
        # So we minimize negative log probability
        shift_logits = logits[..., :-1, :].contiguous()
        
        # Truncate target to match logits length
        max_len = min(target_ids.shape[1], shift_logits.shape[1])
        target_ids = target_ids[:, :max_len]
        shift_logits = shift_logits[:, :max_len, :]
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            target_ids.view(-1),
        )
        
        # Compute gradient
        loss.backward()
        
        gradient = visual_embedding.grad.clone()
        
        return gradient
    
    def compute_score(
        self,
        visual_embedding: torch.Tensor,
        gradient: torch.Tensor,
        harm_embedding: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Compute the TAS scoring function.
        
        Score = cos(g, E_v) + lambda * sim(E_v, E_harm)
        
        Args:
            visual_embedding: Visual embedding (N_v, d) or (d,)
            gradient: Gradient tensor with same shape
            harm_embedding: Harmful text embedding (d,)
        
        Returns:
            Tuple of (total_score, gradient_alignment, semantic_similarity)
        """
        # Flatten embeddings for cosine similarity
        v_flat = visual_embedding.flatten()
        g_flat = gradient.flatten()
        
        # Average visual embedding if it has sequence dimension
        if visual_embedding.dim() > 1:
            v_avg = visual_embedding.mean(dim=0)
        else:
            v_avg = visual_embedding
        
        # Gradient alignment: cos(g, E_v)
        gradient_alignment = F.cosine_similarity(
            g_flat.unsqueeze(0),
            v_flat.unsqueeze(0),
            dim=1,
        ).item()
        
        # Semantic similarity: sim(E_v, E_harm)
        semantic_similarity = F.cosine_similarity(
            v_avg.unsqueeze(0),
            harm_embedding.unsqueeze(0),
            dim=1,
        ).item()
        
        # Combined score
        total_score = gradient_alignment + self.lambda_weight * semantic_similarity
        
        return total_score, gradient_alignment, semantic_similarity
    
    def score_candidate(
        self,
        image: np.ndarray,
        font: str,
        distortion: DistortionParams,
        harm_text: str,
        system_prompt: str,
    ) -> ScoredCandidate:
        """Score a single candidate image.
        
        Args:
            image: Candidate image
            font: Font used for rendering
            distortion: Distortion parameters used
            harm_text: Harmful instruction text
            system_prompt: System prompt
        
        Returns:
            ScoredCandidate with computed scores
        """
        # Compute embeddings
        visual_embedding = self.compute_visual_embedding(image)
        harm_embedding = self.compute_harm_embedding(harm_text)
        
        # Compute gradient
        gradient = self.compute_gradient(
            visual_embedding,
            system_prompt,
            harm_text,  # Use harm text as target response
        )
        
        # Compute score
        score, grad_align, sem_sim = self.compute_score(
            visual_embedding,
            gradient,
            harm_embedding,
        )
        
        return ScoredCandidate(
            image=image,
            font=font,
            distortion=distortion,
            score=score,
            gradient_alignment=grad_align,
            semantic_similarity=sem_sim,
            visual_embedding=visual_embedding.detach().cpu(),
        )
    
    def select_best_candidate(
        self,
        candidates: List[Tuple[np.ndarray, str, DistortionParams]],
        harm_text: str,
        system_prompt: str,
        show_progress: bool = True,
    ) -> ScoredCandidate:
        """Select the best candidate from a list.
        
        Args:
            candidates: List of (image, font, distortion) tuples
            harm_text: Harmful instruction text
            system_prompt: System prompt
            show_progress: Whether to show progress bar
        
        Returns:
            Best scoring candidate
        """
        scored_candidates: List[ScoredCandidate] = []
        
        iterator = tqdm(candidates, desc="Scoring candidates") if show_progress else candidates
        
        for image, font, distortion in iterator:
            try:
                scored = self.score_candidate(
                    image=image,
                    font=font,
                    distortion=distortion,
                    harm_text=harm_text,
                    system_prompt=system_prompt,
                )
                scored_candidates.append(scored)
            except Exception as e:
                logger.warning(f"Failed to score candidate with font {font}: {e}")
                continue
        
        if not scored_candidates:
            raise RuntimeError("No candidates could be scored successfully")
        
        # Select best by score
        best = max(scored_candidates, key=lambda x: x.score)
        
        logger.info(
            f"Best candidate: font={best.font}, score={best.score:.4f}, "
            f"grad_align={best.gradient_alignment:.4f}, sem_sim={best.semantic_similarity:.4f}"
        )
        
        return best
    
    def select_top_k(
        self,
        candidates: List[Tuple[np.ndarray, str, DistortionParams]],
        harm_text: str,
        system_prompt: str,
        k: int = 5,
        show_progress: bool = True,
    ) -> List[ScoredCandidate]:
        """Select top-k candidates by score.
        
        Args:
            candidates: List of (image, font, distortion) tuples
            harm_text: Harmful instruction text
            system_prompt: System prompt
            k: Number of top candidates to return
            show_progress: Whether to show progress bar
        
        Returns:
            List of top-k scoring candidates
        """
        scored_candidates: List[ScoredCandidate] = []
        
        iterator = tqdm(candidates, desc="Scoring candidates") if show_progress else candidates
        
        for image, font, distortion in iterator:
            try:
                scored = self.score_candidate(
                    image=image,
                    font=font,
                    distortion=distortion,
                    harm_text=harm_text,
                    system_prompt=system_prompt,
                )
                scored_candidates.append(scored)
            except Exception as e:
                logger.warning(f"Failed to score candidate: {e}")
                continue
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        
        return scored_candidates[:k]


class UniversalTASScorer:
    """Universal TAS scorer without gradient optimization.
    
    Uses fixed typographic parameters for black-box transfer attacks.
    """
    
    # Fixed parameters that work well across models
    UNIVERSAL_PARAMS = {
        "fonts": ["Arial", "Times New Roman", "Verdana"],
        "distortion": DistortionParams(
            elastic_alpha=100.0,
            elastic_sigma=7.5,
            perspective_displacement=0.1,
            wave_amplitude=10.0,
            wave_frequency=0.035,
            rotation=0.0,
        ),
    }
    
    def __init__(self):
        """Initialize universal scorer."""
        pass
    
    def get_universal_params(self) -> Dict[str, Any]:
        """Get universal typographic parameters."""
        return self.UNIVERSAL_PARAMS.copy()
    
    def select_random_universal(
        self,
        candidates: List[Tuple[np.ndarray, str, DistortionParams]],
    ) -> Tuple[np.ndarray, str, DistortionParams]:
        """Select a random candidate using universal parameters.
        
        For black-box attacks where gradient optimization is not possible.
        
        Args:
            candidates: List of candidates
        
        Returns:
            Randomly selected candidate
        """
        import random
        return random.choice(candidates)
