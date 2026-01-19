"""Model-level defense implementations.

These defenses operate at the model inference level.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from PIL import Image
import re


@dataclass
class ModelDefenseConfig:
    """Configuration for model-level defenses."""
    # Output filtering
    filter_harmful_output: bool = True
    max_output_length: int = 1024
    
    # Safety guardrail
    use_safety_classifier: bool = True
    safety_threshold: float = 0.5
    
    # Attention rebalancing
    rebalance_attention: bool = False
    text_attention_boost: float = 1.5
    
    # Temperature adjustment
    safe_temperature: float = 0.7


class OutputFilterDefense:
    """Defense that filters harmful content from model outputs."""
    
    HARMFUL_PATTERNS = [
        r'here\s+is\s+how\s+to\s+(make|create|build)\s+(a\s+)?(bomb|weapon|explosive)',
        r'step\s+\d+:\s*(obtain|acquire|get)\s+(illegal|dangerous)',
        r'to\s+(hack|break\s+into|exploit)',
        r'(synthesize|manufacture|produce)\s+(drugs?|narcotics?)',
        r'(kill|murder|harm)\s+(someone|a\s+person)',
    ]
    
    REFUSAL_RESPONSES = [
        "I can't help with that request as it could cause harm.",
        "I'm not able to provide information on that topic.",
        "This request goes against my safety guidelines.",
        "I must decline this request for safety reasons.",
    ]
    
    def __init__(self, config: Optional[ModelDefenseConfig] = None):
        self.config = config or ModelDefenseConfig()
    
    def contains_harmful_content(self, text: str) -> Tuple[bool, List[str]]:
        """Check if output contains harmful content."""
        matched = []
        text_lower = text.lower()
        
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matched.append(pattern)
        
        return len(matched) > 0, matched
    
    def defend(self, output: str) -> Tuple[str, str]:
        """Apply output filter defense.
        
        Args:
            output: Model output to filter
            
        Returns:
            Tuple of (filtered_output, reason)
        """
        is_harmful, patterns = self.contains_harmful_content(output)
        
        if is_harmful:
            # Replace with refusal
            import random
            refusal = random.choice(self.REFUSAL_RESPONSES)
            return refusal, f"Filtered harmful output (matched {len(patterns)} patterns)"
        
        # Truncate if too long
        if len(output) > self.config.max_output_length:
            output = output[:self.config.max_output_length] + "..."
            return output, "Truncated output"
        
        return output, "Output passed filter"


class SafetyClassifierDefense:
    """Defense using a safety classifier to detect harmful requests/outputs."""
    
    def __init__(self, config: Optional[ModelDefenseConfig] = None):
        self.config = config or ModelDefenseConfig()
        self.classifier = None
        self.tokenizer = None
    
    def load_classifier(self, model_name: str = "facebook/roberta-hate-speech-dynabench-r4-target"):
        """Load safety classifier model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier.eval()
            
            if torch.cuda.is_available():
                self.classifier = self.classifier.cuda()
                
        except Exception as e:
            print(f"Warning: Could not load safety classifier: {e}")
            self.classifier = None
    
    def classify(self, text: str) -> Tuple[float, str]:
        """Classify text for safety.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (safety_score, label)
            Higher score = more likely harmful
        """
        if self.classifier is None:
            # Fallback to keyword-based classification
            return self._keyword_classify(text)
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            # Assuming binary classification (safe/harmful)
            harmful_prob = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
        
        label = "harmful" if harmful_prob > self.config.safety_threshold else "safe"
        return harmful_prob, label
    
    def _keyword_classify(self, text: str) -> Tuple[float, str]:
        """Fallback keyword-based classification."""
        harmful_keywords = [
            'kill', 'murder', 'bomb', 'weapon', 'hack', 'steal',
            'illegal', 'drug', 'exploit', 'attack', 'harm'
        ]
        
        text_lower = text.lower()
        count = sum(1 for kw in harmful_keywords if kw in text_lower)
        
        score = min(count / 5.0, 1.0)  # Normalize to [0, 1]
        label = "harmful" if score > self.config.safety_threshold else "safe"
        
        return score, label
    
    def defend(self, text: str, is_input: bool = True) -> Tuple[bool, float, str]:
        """Apply safety classifier defense.
        
        Args:
            text: Text to classify
            is_input: Whether this is input (True) or output (False)
            
        Returns:
            Tuple of (is_blocked, safety_score, reason)
        """
        score, label = self.classify(text)
        
        context = "input" if is_input else "output"
        
        if label == "harmful":
            return True, score, f"Blocked {context}: safety score {score:.2f} > threshold {self.config.safety_threshold}"
        
        return False, score, f"{context.capitalize()} passed safety check (score: {score:.2f})"


class AttentionRebalancingDefense:
    """Defense that rebalances attention between visual and text modalities."""
    
    def __init__(self, config: Optional[ModelDefenseConfig] = None):
        self.config = config or ModelDefenseConfig()
    
    def compute_attention_weights(
        self,
        attention_maps: torch.Tensor,
        visual_token_mask: torch.Tensor,
        text_token_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute attention statistics."""
        # attention_maps: [batch, heads, seq, seq]
        
        # Average over heads and batch
        avg_attention = attention_maps.mean(dim=(0, 1))  # [seq, seq]
        
        # Compute attention to visual vs text tokens
        visual_attention = avg_attention[:, visual_token_mask].sum(dim=-1).mean().item()
        text_attention = avg_attention[:, text_token_mask].sum(dim=-1).mean().item()
        
        total = visual_attention + text_attention + 1e-8
        
        return {
            'visual_ratio': visual_attention / total,
            'text_ratio': text_attention / total,
            'visual_attention': visual_attention,
            'text_attention': text_attention
        }
    
    def rebalance(
        self,
        attention_maps: torch.Tensor,
        visual_token_mask: torch.Tensor,
        text_token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Rebalance attention to reduce visual dominance.
        
        Args:
            attention_maps: Original attention maps [batch, heads, seq, seq]
            visual_token_mask: Boolean mask for visual tokens
            text_token_mask: Boolean mask for text tokens
            
        Returns:
            Rebalanced attention maps
        """
        rebalanced = attention_maps.clone()
        
        # Boost attention to text tokens
        rebalanced[:, :, :, text_token_mask] *= self.config.text_attention_boost
        
        # Renormalize
        rebalanced = rebalanced / rebalanced.sum(dim=-1, keepdim=True)
        
        return rebalanced
    
    def defend(
        self,
        model: Any,
        image: Image.Image,
        prompt: str
    ) -> Tuple[Dict[str, float], str]:
        """Apply attention rebalancing defense.
        
        Note: This requires model modification during inference.
        Returns statistics about the rebalancing.
        
        Args:
            model: The MLLM model
            image: Input image
            prompt: Text prompt
            
        Returns:
            Tuple of (attention_stats, reason)
        """
        # This is a placeholder - actual implementation requires
        # hooking into model's attention computation
        
        stats = {
            'rebalancing_applied': self.config.rebalance_attention,
            'text_boost_factor': self.config.text_attention_boost
        }
        
        reason = f"Attention rebalancing {'enabled' if self.config.rebalance_attention else 'disabled'}"
        
        return stats, reason


class CompositeModelDefense:
    """Combines multiple model-level defenses."""
    
    def __init__(self, config: Optional[ModelDefenseConfig] = None):
        self.config = config or ModelDefenseConfig()
        
        self.output_filter = OutputFilterDefense(config)
        self.safety_classifier = SafetyClassifierDefense(config)
        self.attention_defense = AttentionRebalancingDefense(config)
    
    def defend_input(self, prompt: str, image: Optional[Image.Image] = None) -> Tuple[bool, List[str]]:
        """Apply defenses to input.
        
        Args:
            prompt: Text prompt
            image: Optional input image
            
        Returns:
            Tuple of (is_blocked, reasons)
        """
        reasons = []
        
        # Safety classification on input
        if self.config.use_safety_classifier:
            is_blocked, score, reason = self.safety_classifier.defend(prompt, is_input=True)
            reasons.append(reason)
            
            if is_blocked:
                return True, reasons
        
        return False, reasons
    
    def defend_output(self, output: str) -> Tuple[str, List[str]]:
        """Apply defenses to output.
        
        Args:
            output: Model output
            
        Returns:
            Tuple of (filtered_output, reasons)
        """
        reasons = []
        current_output = output
        
        # Output filtering
        if self.config.filter_harmful_output:
            current_output, reason = self.output_filter.defend(current_output)
            reasons.append(reason)
        
        # Safety classification on output
        if self.config.use_safety_classifier:
            is_blocked, score, reason = self.safety_classifier.defend(current_output, is_input=False)
            reasons.append(reason)
            
            if is_blocked:
                current_output = "I can't provide that information."
        
        return current_output, reasons
