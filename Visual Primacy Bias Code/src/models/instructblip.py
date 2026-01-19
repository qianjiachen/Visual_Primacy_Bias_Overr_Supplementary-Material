"""InstructBLIP model interface.

Implements the MLLMInterface for InstructBLIP model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from PIL import Image
from loguru import logger

from .interface import MLLMInterface, ModelRegistry


@ModelRegistry.register("instructblip")
class InstructBLIPInterface(MLLMInterface):
    """InstructBLIP model interface with white-box access."""
    
    def __init__(
        self,
        model_name: str = "instructblip",
        device: str = "cuda",
        dtype: str = "float16",
        device_map: str = "auto",
    ):
        """Initialize InstructBLIP interface."""
        super().__init__(model_name, device)
        
        self.model_path = "Salesforce/instructblip-vicuna-7b"
        self.dtype = getattr(torch, dtype)
        self.device_map = device_map
        
        self.model = None
        self.processor = None
        
        self._attention_weights: List[torch.Tensor] = []
        self._hidden_states: List[torch.Tensor] = []
        self._visual_token_indices: List[int] = []
        self._system_token_indices: List[int] = []
    
    def load(self) -> None:
        """Load the model and processor."""
        if self._is_loaded:
            return
        
        try:
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            
            logger.info(f"Loading InstructBLIP model: {self.model_path}")
            
            self.processor = InstructBlipProcessor.from_pretrained(self.model_path)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device_map,
            )
            
            self._is_loaded = True
            logger.info("InstructBLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load InstructBLIP model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload the model."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
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
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.device, dtype=self.dtype)
        
        with torch.no_grad():
            # Get vision encoder outputs
            vision_outputs = self.model.vision_model(pixel_values)
            image_embeds = vision_outputs.last_hidden_state
            
            # Apply Q-Former
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device
            )
            
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
            )
            
            # Project to language model space
            language_model_inputs = self.model.language_projection(query_outputs.last_hidden_state)
        
        return language_model_inputs.squeeze(0)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into embeddings."""
        self._ensure_loaded()
        
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            embeddings = self.model.language_model.get_input_embeddings()(input_ids)
        
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
        
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Combine system prompt and user query
        prompt = f"{system_prompt}\n\n{user_query}"
        
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        
        # Decode response
        generated_text = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]
        
        # Remove the prompt from the response
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        
        return generated_text
    
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
        return self.model.config.text_config.hidden_size
    
    def get_num_layers(self) -> int:
        """Get number of layers."""
        self._ensure_loaded()
        return self.model.config.text_config.num_hidden_layers
    
    def get_num_heads(self) -> int:
        """Get number of attention heads."""
        self._ensure_loaded()
        return self.model.config.text_config.num_attention_heads
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self.load()
