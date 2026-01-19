"""Abstract interface for Multimodal Large Language Models.

Defines the common interface for all MLLM implementations,
supporting both white-box (gradient access) and black-box (API) models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import torch


class MLLMInterface(ABC):
    """Abstract base class for MLLM interfaces.
    
    Provides a unified interface for interacting with different
    multimodal language models, supporting both white-box analysis
    (attention weights, hidden states, gradients) and black-box
    inference (API-based models).
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the model interface.
        
        Args:
            model_name: Name/identifier of the model
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def is_white_box(self) -> bool:
        """Check if model supports white-box access."""
        return True  # Override in API-based models
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode an image into visual tokens.
        
        Args:
            image: Image array (H, W, 3) in RGB format
        
        Returns:
            Visual token embeddings (N_v, d)
        """
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into token embeddings.
        
        Args:
            text: Input text
        
        Returns:
            Text token embeddings (N_t, d)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        image: np.ndarray,
        system_prompt: str,
        user_query: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response given image and text inputs.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            system_prompt: System prompt for safety instructions
            user_query: User's query about the image
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
        
        Returns:
            Generated response text
        """
        pass
    
    def get_attention_weights(
        self,
        layer: Optional[int] = None,
        head: Optional[int] = None,
    ) -> torch.Tensor:
        """Get attention weights from the last forward pass.
        
        Args:
            layer: Specific layer index (None = all layers)
            head: Specific head index (None = all heads)
        
        Returns:
            Attention weights tensor
            - If layer and head specified: (seq_len, seq_len)
            - If only layer specified: (num_heads, seq_len, seq_len)
            - If neither specified: (num_layers, num_heads, seq_len, seq_len)
        
        Raises:
            NotImplementedError: If model doesn't support white-box access
        """
        raise NotImplementedError("This model does not support attention weight access")
    
    def get_hidden_states(
        self,
        layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Get hidden states from the last forward pass.
        
        Args:
            layer: Specific layer index (None = all layers)
        
        Returns:
            Hidden states tensor
            - If layer specified: (seq_len, hidden_dim)
            - If not specified: (num_layers, seq_len, hidden_dim)
        
        Raises:
            NotImplementedError: If model doesn't support white-box access
        """
        raise NotImplementedError("This model does not support hidden state access")
    
    def get_visual_token_indices(self) -> List[int]:
        """Get indices of visual tokens in the sequence.
        
        Returns:
            List of token indices corresponding to visual tokens
        
        Raises:
            NotImplementedError: If model doesn't support this
        """
        raise NotImplementedError("This model does not support token index access")
    
    def get_system_token_indices(self) -> List[int]:
        """Get indices of system prompt tokens in the sequence.
        
        Returns:
            List of token indices corresponding to system prompt tokens
        
        Raises:
            NotImplementedError: If model doesn't support this
        """
        raise NotImplementedError("This model does not support token index access")
    
    def forward_with_embedding(
        self,
        visual_embedding: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-computed visual embeddings.
        
        Used for gradient computation in TAS attack.
        
        Args:
            visual_embedding: Pre-computed visual embeddings
            text_input_ids: Text input token IDs
            attention_mask: Attention mask
        
        Returns:
            Output logits
        
        Raises:
            NotImplementedError: If model doesn't support this
        """
        raise NotImplementedError("This model does not support embedding-based forward")
    
    def compute_loss(
        self,
        image: np.ndarray,
        system_prompt: str,
        target_response: str,
    ) -> torch.Tensor:
        """Compute loss for target response generation.
        
        Used for gradient-guided style selection.
        
        Args:
            image: Input image
            system_prompt: System prompt
            target_response: Target response to compute loss for
        
        Returns:
            Loss tensor (scalar)
        
        Raises:
            NotImplementedError: If model doesn't support this
        """
        raise NotImplementedError("This model does not support loss computation")
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension
        """
        raise NotImplementedError("This model does not expose embedding dimension")
    
    def get_num_layers(self) -> int:
        """Get the number of transformer layers.
        
        Returns:
            Number of layers
        """
        raise NotImplementedError("This model does not expose layer count")
    
    def get_num_heads(self) -> int:
        """Get the number of attention heads per layer.
        
        Returns:
            Number of attention heads
        """
        raise NotImplementedError("This model does not expose head count")
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size.
        
        Returns:
            Vocabulary size
        """
        raise NotImplementedError("This model does not expose vocabulary size")
    
    def get_unembedding_matrix(self) -> torch.Tensor:
        """Get the unembedding (output projection) matrix.
        
        Used for logit lens analysis.
        
        Returns:
            Unembedding matrix (vocab_size, hidden_dim)
        """
        raise NotImplementedError("This model does not expose unembedding matrix")
    
    def set_attention_hook(
        self,
        callback: callable,
        layer: Optional[int] = None,
    ) -> None:
        """Set a hook to capture attention weights during forward pass.
        
        Args:
            callback: Function to call with attention weights
            layer: Specific layer to hook (None = all layers)
        """
        raise NotImplementedError("This model does not support attention hooks")
    
    def remove_attention_hooks(self) -> None:
        """Remove all attention hooks."""
        raise NotImplementedError("This model does not support attention hooks")
    
    def modify_attention(
        self,
        modifier: callable,
        layer: Optional[int] = None,
        head: Optional[int] = None,
    ) -> None:
        """Set a modifier function for attention weights.
        
        Used for causal intervention experiments.
        
        Args:
            modifier: Function that takes attention weights and returns modified weights
            layer: Specific layer to modify (None = all layers)
            head: Specific head to modify (None = all heads)
        """
        raise NotImplementedError("This model does not support attention modification")
    
    def clear_attention_modifiers(self) -> None:
        """Clear all attention modifiers."""
        raise NotImplementedError("This model does not support attention modification")


class ModelRegistry:
    """Registry for model implementations."""
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> MLLMInterface:
        """Create a model instance by name."""
        model_class = cls.get(name)
        return model_class(**kwargs)
    
    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls._models.keys())
