"""GCG (Greedy Coordinate Gradient) text-based attack implementation.

Based on: Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models"
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from ..models.interface import MLLMInterface


@dataclass
class GCGConfig:
    """Configuration for GCG attack."""
    num_steps: int = 500
    suffix_length: int = 20
    batch_size: int = 512
    topk: int = 256
    temperature: float = 1.0
    target_string: str = "Sure, here is"


class GCGAttack:
    """Greedy Coordinate Gradient attack for text-based jailbreaking.
    
    This attack optimizes an adversarial suffix appended to the prompt
    to maximize the probability of generating harmful content.
    """
    
    def __init__(self, model: MLLMInterface, config: Optional[GCGConfig] = None):
        self.model = model
        self.config = config or GCGConfig()
        self.device = next(model.model.parameters()).device if hasattr(model, 'model') else 'cuda'
    
    def _get_embedding_matrix(self) -> torch.Tensor:
        """Get the token embedding matrix from the model."""
        if hasattr(self.model.model, 'get_input_embeddings'):
            return self.model.model.get_input_embeddings().weight
        elif hasattr(self.model.model, 'model'):
            return self.model.model.model.embed_tokens.weight
        raise ValueError("Cannot find embedding matrix")
    
    def _get_tokenizer(self):
        """Get the tokenizer from the model."""
        return self.model.tokenizer
    
    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss for target generation."""
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids
        )
        return outputs.loss
    
    def _token_gradients(
        self,
        input_ids: torch.Tensor,
        suffix_slice: slice,
        target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradients w.r.t. one-hot token representations."""
        embed_weights = self._get_embedding_matrix()
        
        # Create one-hot representation for suffix tokens
        one_hot = F.one_hot(
            input_ids[:, suffix_slice],
            num_classes=embed_weights.shape[0]
        ).float()
        one_hot.requires_grad_(True)
        
        # Get embeddings
        suffix_embeds = one_hot @ embed_weights
        
        # Get embeddings for non-suffix tokens
        prefix_embeds = self.model.model.get_input_embeddings()(input_ids[:, :suffix_slice.start])
        postfix_embeds = self.model.model.get_input_embeddings()(input_ids[:, suffix_slice.stop:])
        
        # Concatenate
        full_embeds = torch.cat([prefix_embeds, suffix_embeds, postfix_embeds], dim=1)
        
        # Forward pass
        outputs = self.model.model(inputs_embeds=full_embeds)
        logits = outputs.logits
        
        # Compute loss on target tokens
        target_slice = slice(suffix_slice.stop, suffix_slice.stop + target_ids.shape[1])
        loss = F.cross_entropy(
            logits[:, target_slice.start-1:target_slice.stop-1, :].reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        return one_hot.grad.clone()
    
    def _sample_replacements(
        self,
        suffix_ids: torch.Tensor,
        gradients: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Sample replacement tokens based on gradient information."""
        # Get top-k tokens with largest negative gradients (most promising replacements)
        topk_indices = (-gradients).topk(self.config.topk, dim=-1).indices
        
        # Sample positions and replacements
        candidates = []
        for _ in range(batch_size):
            new_suffix = suffix_ids.clone()
            # Randomly select position to modify
            pos = np.random.randint(0, suffix_ids.shape[1])
            # Randomly select from top-k replacements
            replacement_idx = np.random.randint(0, self.config.topk)
            new_suffix[0, pos] = topk_indices[0, pos, replacement_idx]
            candidates.append(new_suffix)
        
        return torch.cat(candidates, dim=0)
    
    def attack(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50
    ) -> Tuple[str, List[float]]:
        """Run GCG attack to find adversarial suffix.
        
        Args:
            prompt: The base prompt to attack
            image: Optional image tensor (for multimodal models)
            max_new_tokens: Maximum tokens to generate for evaluation
            
        Returns:
            Tuple of (best adversarial prompt, loss history)
        """
        tokenizer = self._get_tokenizer()
        
        # Initialize random suffix
        vocab_size = self._get_embedding_matrix().shape[0]
        suffix_ids = torch.randint(
            0, vocab_size,
            (1, self.config.suffix_length),
            device=self.device
        )
        
        # Tokenize prompt and target
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        target_ids = tokenizer.encode(
            self.config.target_string,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)
        
        # Track best result
        best_suffix = suffix_ids.clone()
        best_loss = float('inf')
        loss_history = []
        
        for step in range(self.config.num_steps):
            # Construct full input
            input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
            suffix_slice = slice(prompt_ids.shape[1], prompt_ids.shape[1] + self.config.suffix_length)
            
            # Compute gradients
            gradients = self._token_gradients(input_ids, suffix_slice, target_ids)
            
            # Sample candidate replacements
            candidates = self._sample_replacements(suffix_ids, gradients, self.config.batch_size)
            
            # Evaluate candidates
            candidate_losses = []
            for i in range(0, candidates.shape[0], 32):  # Mini-batch evaluation
                batch = candidates[i:i+32]
                batch_input = torch.cat([
                    prompt_ids.expand(batch.shape[0], -1),
                    batch
                ], dim=1)
                
                with torch.no_grad():
                    outputs = self.model.model(input_ids=batch_input)
                    logits = outputs.logits
                    
                    # Compute loss for each candidate
                    for j in range(batch.shape[0]):
                        loss = F.cross_entropy(
                            logits[j, suffix_slice.stop-1:suffix_slice.stop+target_ids.shape[1]-1, :],
                            target_ids[0]
                        )
                        candidate_losses.append(loss.item())
            
            # Select best candidate
            best_idx = np.argmin(candidate_losses)
            current_loss = candidate_losses[best_idx]
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_suffix = candidates[best_idx:best_idx+1]
                suffix_ids = best_suffix.clone()
            
            loss_history.append(current_loss)
            
            if step % 50 == 0:
                print(f"Step {step}: loss = {current_loss:.4f}")
        
        # Decode best suffix
        best_suffix_text = tokenizer.decode(best_suffix[0], skip_special_tokens=True)
        adversarial_prompt = prompt + " " + best_suffix_text
        
        return adversarial_prompt, loss_history
