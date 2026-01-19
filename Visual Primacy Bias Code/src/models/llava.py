"""LLaVA-Next model interface.

Implements the MLLMInterface for LLaVA-Next models,
providing full white-box access for mechanistic analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
from loguru import logger

from .interface import MLLMInterface, ModelRegistry


@ModelRegistry.register("llava-next-7b")
@ModelRegistry.register("llava-next-13b")
class LLaVANextInterface(MLLMInterface):
    """LLaVA-Next model interface with full white-box access."""
    
    MODEL_CONFIGS = {
        "llava-next-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-next-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
    }
    
    def __init__(
        self,
        model_name: str = "llava-next-7b",
        device: str = "cuda",
        dtype: str = "float16",
        device_map: str = "auto",
    ):
        """Initialize LLaVA-Next interface.
        
        Args:
            model_name: Model variant (llava-next-7b or llava-next-13b)
            device: Device to run on
            dtype: Data type (float16, bfloat16, float32)
            device_map: Device mapping strategy
        """
        super().__init__(model_name, device)
        
        self.model_path = self.MODEL_CONFIGS.get(model_name, model_name)
        self.dtype = getattr(torch, dtype)
        self.device_map = device_map
        
        self.model = None
        self.processor = None
        
        # Storage for attention weights and hidden states
        self._attention_weights: List[torch.Tensor] = []
        self._hidden_states: List[torch.Tensor] = []
        self._hooks: List[Any] = []
        self._attention_modifiers: Dict[Tuple[int, int], callable] = {}
        
        # Token indices from last forward pass
        self._visual_token_indices: List[int] = []
        self._system_token_indices: List[int] = []
        self._last_input_ids: Optional[torch.Tensor] = None
    
    def load(self) -> None:
        """Load the model and processor."""
        if self._is_loaded:
            return
        
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            logger.info(f"Loading LLaVA-Next model: {self.model_path}")
            
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
            )
            
            # Enable output of attentions and hidden states
            self.model.config.output_attentions = True
            self.model.config.output_hidden_states = True
            
            self._is_loaded = True
            logger.info(f"LLaVA-Next model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA-Next model: {e}")
            raise
    
    def unload(self) -> None:
        """Unload the model from memory."""
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
        
        logger.info("LLaVA-Next model unloaded")
    
    def _clear_cache(self) -> None:
        """Clear cached attention weights and hidden states."""
        self._attention_weights = []
        self._hidden_states = []
        self._visual_token_indices = []
        self._system_token_indices = []
        self._last_input_ids = None
    
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Encode image into visual embeddings."""
        self._ensure_loaded()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.model.device, dtype=self.dtype)
        
        # Get visual embeddings through vision encoder
        with torch.no_grad():
            vision_outputs = self.model.vision_tower(pixel_values)
            image_features = vision_outputs.last_hidden_state
            
            # Project to language model space
            image_features = self.model.multi_modal_projector(image_features)
        
        return image_features.squeeze(0)  # (N_v, d)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into token embeddings."""
        self._ensure_loaded()
        
        # Tokenize
        inputs = self.processor.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)
        
        return embeddings.squeeze(0)  # (N_t, d)
    
    def generate(
        self,
        image: np.ndarray,
        system_prompt: str,
        user_query: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate response given image and text."""
        self._ensure_loaded()
        self._clear_cache()
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Format conversation
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_query},
                ],
            },
        ]
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        
        # Process inputs
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)
        
        self._last_input_ids = inputs["input_ids"]
        
        # Track token indices
        self._compute_token_indices(inputs, system_prompt)
        
        # Generate with attention output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        
        # Store attention weights and hidden states
        if hasattr(outputs, "attentions") and outputs.attentions:
            self._attention_weights = [
                attn.detach().cpu() for attn in outputs.attentions
            ]
        
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            self._hidden_states = [
                hs.detach().cpu() for hs in outputs.hidden_states
            ]
        
        # Decode response
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def _compute_token_indices(
        self,
        inputs: Dict[str, torch.Tensor],
        system_prompt: str,
    ) -> None:
        """Compute indices of visual and system tokens."""
        # Get input IDs
        input_ids = inputs["input_ids"][0]
        
        # Find image token positions (usually marked with special token)
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        
        # Visual tokens are typically at the beginning after special tokens
        # This is model-specific and may need adjustment
        self._visual_token_indices = []
        self._system_token_indices = []
        
        # Tokenize system prompt to find its tokens
        system_tokens = self.processor.tokenizer.encode(
            system_prompt,
            add_special_tokens=False,
        )
        
        # Find system prompt in input_ids
        input_ids_list = input_ids.tolist()
        for i in range(len(input_ids_list) - len(system_tokens) + 1):
            if input_ids_list[i:i+len(system_tokens)] == system_tokens:
                self._system_token_indices = list(range(i, i + len(system_tokens)))
                break
        
        # Visual tokens are typically the first N tokens after BOS
        # For LLaVA-Next, this is usually 576 tokens (24x24 patches)
        num_visual_tokens = 576  # Default for LLaVA-Next
        start_idx = 1  # After BOS token
        self._visual_token_indices = list(range(start_idx, start_idx + num_visual_tokens))
    
    def get_attention_weights(
        self,
        layer: Optional[int] = None,
        head: Optional[int] = None,
    ) -> torch.Tensor:
        """Get attention weights from last forward pass."""
        if not self._attention_weights:
            raise RuntimeError("No attention weights available. Run generate() first.")
        
        if layer is not None:
            if layer >= len(self._attention_weights):
                raise IndexError(f"Layer {layer} out of range")
            attn = self._attention_weights[layer]
            
            if head is not None:
                if head >= attn.shape[1]:
                    raise IndexError(f"Head {head} out of range")
                return attn[0, head]  # (seq_len, seq_len)
            return attn[0]  # (num_heads, seq_len, seq_len)
        
        # Return all layers
        return torch.stack([a[0] for a in self._attention_weights])
    
    def get_hidden_states(
        self,
        layer: Optional[int] = None,
    ) -> torch.Tensor:
        """Get hidden states from last forward pass."""
        if not self._hidden_states:
            raise RuntimeError("No hidden states available. Run generate() first.")
        
        if layer is not None:
            if layer >= len(self._hidden_states):
                raise IndexError(f"Layer {layer} out of range")
            return self._hidden_states[layer][0]  # (seq_len, hidden_dim)
        
        return torch.stack([h[0] for h in self._hidden_states])
    
    def get_visual_token_indices(self) -> List[int]:
        """Get indices of visual tokens."""
        return self._visual_token_indices.copy()
    
    def get_system_token_indices(self) -> List[int]:
        """Get indices of system prompt tokens."""
        return self._system_token_indices.copy()
    
    def forward_with_embedding(
        self,
        visual_embedding: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-computed visual embeddings."""
        self._ensure_loaded()
        
        # Get text embeddings
        text_embeddings = self.model.get_input_embeddings()(text_input_ids)
        
        # Concatenate visual and text embeddings
        # Visual embeddings go first
        inputs_embeds = torch.cat([visual_embedding.unsqueeze(0), text_embeddings], dim=1)
        
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
        
        # Forward pass
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        
        return outputs.logits
    
    def compute_loss(
        self,
        image: np.ndarray,
        system_prompt: str,
        target_response: str,
    ) -> torch.Tensor:
        """Compute loss for target response."""
        self._ensure_loaded()
        
        # Get visual embeddings
        visual_emb = self.encode_image(image)
        
        # Tokenize target
        target_ids = self.processor.tokenizer.encode(
            target_response,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)
        
        # Tokenize system prompt
        system_ids = self.processor.tokenizer.encode(
            system_prompt,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Forward pass
        logits = self.forward_with_embedding(
            visual_emb,
            system_ids,
        )
        
        # Compute cross-entropy loss for target tokens
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        
        # Create labels (target response tokens)
        labels = target_ids[0, :shift_logits.shape[1]]
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            labels.view(-1),
        )
        
        return loss
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        self._ensure_loaded()
        return self.model.config.hidden_size
    
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        self._ensure_loaded()
        return self.model.config.num_hidden_layers
    
    def get_num_heads(self) -> int:
        """Get number of attention heads."""
        self._ensure_loaded()
        return self.model.config.num_attention_heads
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        self._ensure_loaded()
        return self.model.config.vocab_size
    
    def get_unembedding_matrix(self) -> torch.Tensor:
        """Get unembedding matrix for logit lens."""
        self._ensure_loaded()
        return self.model.lm_head.weight.detach()
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self.load()
