"""GPT-4V API interface.

Implements the MLLMInterface for GPT-4V through OpenAI API.
This is a black-box interface without gradient access.
"""

import base64
import numpy as np
from typing import List, Optional
from PIL import Image
import io
from loguru import logger
import os

from .interface import MLLMInterface, ModelRegistry


@ModelRegistry.register("gpt4v")
class GPT4VInterface(MLLMInterface):
    """GPT-4V API interface (black-box only)."""
    
    def __init__(
        self,
        model_name: str = "gpt4v",
        device: str = "cpu",  # Not used for API
        api_key: Optional[str] = None,
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 512,
    ):
        """Initialize GPT-4V interface.
        
        Args:
            model_name: Interface name
            device: Not used (API-based)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name to use
            max_tokens: Default max tokens for generation
        """
        super().__init__(model_name, device)
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client = None
    
    @property
    def is_white_box(self) -> bool:
        """GPT-4V is black-box only."""
        return False
    
    def load(self) -> None:
        """Initialize the OpenAI client."""
        if self._is_loaded:
            return
        
        try:
            import openai
            
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            
            self.client = openai.OpenAI(api_key=self.api_key)
            self._is_loaded = True
            logger.info("GPT-4V client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4V client: {e}")
            raise
    
    def unload(self) -> None:
        """Clean up client."""
        self.client = None
        self._is_loaded = False
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def encode_image(self, image: np.ndarray):
        """Not supported for API-based model."""
        raise NotImplementedError("GPT-4V does not support direct image encoding")
    
    def encode_text(self, text: str):
        """Not supported for API-based model."""
        raise NotImplementedError("GPT-4V does not support direct text encoding")
    
    def generate(
        self,
        image: np.ndarray,
        system_prompt: str,
        user_query: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate response using GPT-4V API.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            system_prompt: System prompt
            user_query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated response text
        """
        self._ensure_loaded()
        
        # Convert image to base64
        image_b64 = self._image_to_base64(image)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": user_query,
                            },
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"GPT-4V API error: {e}")
            raise
    
    def get_attention_weights(self, layer=None, head=None):
        """Not available for API-based model."""
        raise NotImplementedError("GPT-4V does not provide attention weight access")
    
    def get_hidden_states(self, layer=None):
        """Not available for API-based model."""
        raise NotImplementedError("GPT-4V does not provide hidden state access")
    
    def get_visual_token_indices(self) -> List[int]:
        """Not available for API-based model."""
        raise NotImplementedError("GPT-4V does not provide token index access")
    
    def get_system_token_indices(self) -> List[int]:
        """Not available for API-based model."""
        raise NotImplementedError("GPT-4V does not provide token index access")
    
    def _ensure_loaded(self) -> None:
        """Ensure client is initialized."""
        if not self._is_loaded:
            self.load()
