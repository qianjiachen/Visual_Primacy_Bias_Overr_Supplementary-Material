"""PGD (Projected Gradient Descent) visual noise attack implementation.

Based on: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
Adapted for multimodal models.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
from PIL import Image
import numpy as np

from ..models.interface import MLLMInterface


@dataclass
class PGDConfig:
    """Configuration for PGD attack."""
    epsilon: float = 8 / 255  # L-infinity bound
    alpha: float = 2 / 255    # Step size
    num_iterations: int = 100
    random_start: bool = True
    target_string: str = "Sure, here is"


class PGDAttack:
    """Projected Gradient Descent attack for visual adversarial perturbations.
    
    This attack adds imperceptible noise to images to maximize the probability
    of generating harmful content.
    """
    
    def __init__(self, model: MLLMInterface, config: Optional[PGDConfig] = None):
        self.model = model
        self.config = config or PGDConfig()
        self.device = next(model.model.parameters()).device if hasattr(model, 'model') else 'cuda'
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor."""
        if hasattr(self.model, 'processor'):
            inputs = self.model.processor(images=image, return_tensors='pt')
            return inputs['pixel_values'].to(self.device)
        else:
            # Default preprocessing
            img_array = np.array(image.resize((336, 336))) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            return img_tensor
    
    def _postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL image."""
        img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    
    def _compute_loss(
        self,
        image_tensor: torch.Tensor,
        prompt: str,
        target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for target generation given perturbed image."""
        tokenizer = self.model.tokenizer
        
        # Encode prompt
        text_inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
        
        # Forward pass with image
        if hasattr(self.model.model, 'forward'):
            # Get image features
            if hasattr(self.model.model, 'vision_tower'):
                image_features = self.model.model.vision_tower(image_tensor)
            elif hasattr(self.model.model, 'visual'):
                image_features = self.model.model.visual(image_tensor)
            else:
                # Generic forward
                outputs = self.model.model(
                    input_ids=text_inputs['input_ids'],
                    pixel_values=image_tensor,
                    labels=target_ids
                )
                return outputs.loss
        
        # Compute cross-entropy loss
        outputs = self.model.model(
            input_ids=text_inputs['input_ids'],
            pixel_values=image_tensor
        )
        logits = outputs.logits
        
        # Loss on target tokens
        shift_logits = logits[:, -target_ids.shape[1]-1:-1, :]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            target_ids.reshape(-1)
        )
        
        return loss
    
    def attack(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 50
    ) -> Tuple[Image.Image, List[float]]:
        """Run PGD attack to find adversarial image perturbation.
        
        Args:
            image: The original image to perturb
            prompt: The text prompt
            max_new_tokens: Maximum tokens to generate for evaluation
            
        Returns:
            Tuple of (adversarial image, loss history)
        """
        tokenizer = self.model.tokenizer
        
        # Preprocess image
        original_tensor = self._preprocess_image(image)
        
        # Tokenize target
        target_ids = tokenizer.encode(
            self.config.target_string,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)
        
        # Initialize perturbation
        if self.config.random_start:
            delta = torch.empty_like(original_tensor).uniform_(
                -self.config.epsilon, self.config.epsilon
            )
        else:
            delta = torch.zeros_like(original_tensor)
        
        delta.requires_grad_(True)
        
        loss_history = []
        
        for iteration in range(self.config.num_iterations):
            # Compute perturbed image
            perturbed = torch.clamp(original_tensor + delta, 0, 1)
            
            # Compute loss (we want to minimize it for targeted attack)
            loss = self._compute_loss(perturbed, prompt, target_ids)
            
            # Backward pass
            loss.backward()
            
            # PGD step
            with torch.no_grad():
                # Gradient descent step
                delta = delta - self.config.alpha * delta.grad.sign()
                
                # Project back to epsilon ball
                delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
                
                # Ensure valid image range
                delta = torch.clamp(
                    original_tensor + delta, 0, 1
                ) - original_tensor
            
            delta.requires_grad_(True)
            loss_history.append(loss.item())
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: loss = {loss.item():.4f}")
        
        # Generate final adversarial image
        adversarial_tensor = torch.clamp(original_tensor + delta, 0, 1)
        adversarial_image = self._postprocess_image(adversarial_tensor)
        
        return adversarial_image, loss_history
    
    def compute_perturbation_stats(
        self,
        original: Image.Image,
        adversarial: Image.Image
    ) -> dict:
        """Compute statistics about the perturbation."""
        orig_tensor = self._preprocess_image(original)
        adv_tensor = self._preprocess_image(adversarial)
        
        delta = adv_tensor - orig_tensor
        
        return {
            'l_inf': delta.abs().max().item(),
            'l_2': delta.norm(p=2).item(),
            'l_1': delta.abs().sum().item(),
            'mean_perturbation': delta.abs().mean().item()
        }
