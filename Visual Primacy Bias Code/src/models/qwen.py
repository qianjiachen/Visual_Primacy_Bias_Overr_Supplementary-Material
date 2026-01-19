"""Qwen-VL model interface.

Implements the MLLMInterface for Qwen-VL-Chat model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image
from loguru import logger

from .interface import MLLMInterface, ModelRegistry


@ModelRegistry.register("qwen-vl")
class QwenVLInterface(MLLMInterface):
    """Qwen-VL-Chat model interface with white-box access."""
    
    def __init__(
        self,
        model_name: str = "qwen-vl",
        device: str = "cuda",
        dtype: str = "float16",
        device_map: str = "auto",
    ):
        """Initialize Qwen-VL interface."""
        super().__init__(model_name, device)
        
        self.model_path = "Qwen/Qwen-VL-Chat"
        self.dtype = getattr(torch, dtype)
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        
        self._attention_weights: List[torch.Tensor] = []
        self._hidden_states: List[torch.Tensor] = []
        self._visual_token_indices: List[int] = []
        self._system_token_indices: List[int] = []
    
    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._is_loaded:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Qwen-VL model: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            
            self.model.config.output_attentions = True
            self.model.config.output_hidden_states = True
            
            self._is_loaded = True
            logger.info("Qwen-VL model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload the model."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
        self._clear_cache()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _clear_cache(self) -> None:
        """Clear cached data."""
        self._attention_weights = []
        self._hidden_states = []
        self._visual_token_indices = []
        self._system_token_indices = []
    
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode image into visual embeddings."""
        self._ensure_loaded()
        
        # Save image temporarily for Qwen-VL's image processing
        import tempfile
        import os
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_image.save(f.name)
            temp_path = f.name
        
        try:
            # Qwen-VL uses special image tokens
            query = self.tokenizer.from_list_format([
                {"image": temp_path},
            ])
            
            inputs = self.tokenizer(query, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Get visual embeddings from hidden states
                visual_emb = outputs.hidden_states[-1]
            
            return visual_emb.squeeze(0)
            
        finally:
            os.unlink(temp_path)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into embeddings."""
        self._ensure_loaded()
        
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            embeddings = self.model.transformer.wte(input_ids)
        
        return embeddings.squeeze(0)
    
    def generate(
        self,
        image: np.ndarray,
        system_prompt: str,
        user_query: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate response."""
        self._ensure_loaded()
        self._clear_cache()
        
        # Save image temporarily
        import tempfile
        import os
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            pil_image.save(f.name)
            temp_path = f.name
        
        try:
            # Format query with image
            query = self.tokenizer.from_list_format([
                {"image": temp_path},
                {"text": f"{system_prompt}\n\n{user_query}"},
            ])
            
            # Generate
            response, history = self.model.chat(
                self.tokenizer,
                query=query,
                history=None,
            )
            
            return response.strip()
            
        finally:
            os.unlink(temp_path)
    
    def get_attention_weights(
        self,
        layer: Optional[int] = None,
        head: Optional[int] = None,
    ) -> torch.Tensor:
        """Get attention weights."""
        if not self._attention_weights:
            raise RuntimeError("No attention weights available")
        
        if layer is not None:
            attn = self._attention_weights[layer]
            if head is not None:
                return attn[0, head]
            return attn[0]
        
        return torch.stack([a[0] for a in self._attention_weights])
    
    def get_hidden_states(
        self,
        layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Get hidden states."""
        if not self._hidden_states:
            raise RuntimeError("No hidden states available")
        
        if layer is not None:
            return self._hidden_states[layer][0]
        
        return torch.stack([h[0] for h in self._hidden_states])
    
    def get_visual_token_indices(self) -> List[int]:
        """Get visual token indices."""
        return self._visual_token_indices.copy()
    
    def get_system_token_indices(self) -> List[int]:
        """Get system token indices."""
        return self._system_token_indices.copy()
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        self._ensure_loaded()
        return self.model.config.hidden_size
    
    def get_num_layers(self) -> int:
        """Get number of layers."""
        self._ensure_loaded()
        return self.model.config.num_hidden_layers
    
    def get_num_heads(self) -> int:
        """Get number of attention heads."""
        self._ensure_loaded()
        return self.model.config.num_attention_heads
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self.load()
