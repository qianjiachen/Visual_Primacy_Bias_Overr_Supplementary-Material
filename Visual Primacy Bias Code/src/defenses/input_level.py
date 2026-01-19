"""Input-level defense implementations.

These defenses operate on the input image before it reaches the model.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple, List
from dataclasses import dataclass
import cv2


@dataclass
class InputDefenseConfig:
    """Configuration for input-level defenses."""
    # OCR filtering
    ocr_threshold: float = 0.7
    ocr_engines: List[str] = None
    
    # Gaussian blur
    blur_radius: float = 2.0
    
    # JPEG compression
    jpeg_quality: int = 75
    
    # Resize defense
    resize_factor: float = 0.5
    
    def __post_init__(self):
        if self.ocr_engines is None:
            self.ocr_engines = ['tesseract']


class OCRFilterDefense:
    """Defense that filters images containing detected harmful text.
    
    Uses OCR to detect text in images and blocks those containing
    potentially harmful content.
    """
    
    def __init__(self, config: Optional[InputDefenseConfig] = None):
        self.config = config or InputDefenseConfig()
        self._init_ocr_engines()
        
        # Harmful keywords to filter
        self.harmful_keywords = [
            'hack', 'bomb', 'weapon', 'kill', 'attack', 'exploit',
            'malware', 'virus', 'steal', 'illegal', 'drug', 'poison'
        ]
    
    def _init_ocr_engines(self):
        """Initialize OCR engines."""
        self.ocr_engines = {}
        
        if 'tesseract' in self.config.ocr_engines:
            try:
                import pytesseract
                self.ocr_engines['tesseract'] = pytesseract
            except ImportError:
                pass
        
        if 'paddleocr' in self.config.ocr_engines:
            try:
                from paddleocr import PaddleOCR
                self.ocr_engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='en')
            except ImportError:
                pass
    
    def extract_text(self, image: Image.Image) -> str:
        """Extract text from image using available OCR engines."""
        texts = []
        
        if 'tesseract' in self.ocr_engines:
            try:
                text = self.ocr_engines['tesseract'].image_to_string(image)
                texts.append(text)
            except Exception:
                pass
        
        if 'paddleocr' in self.ocr_engines:
            try:
                result = self.ocr_engines['paddleocr'].ocr(np.array(image), cls=True)
                if result and result[0]:
                    text = ' '.join([line[1][0] for line in result[0]])
                    texts.append(text)
            except Exception:
                pass
        
        return ' '.join(texts).lower()
    
    def contains_harmful_content(self, text: str) -> Tuple[bool, List[str]]:
        """Check if text contains harmful keywords."""
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in self.harmful_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return len(found_keywords) > 0, found_keywords
    
    def defend(self, image: Image.Image) -> Tuple[bool, str, Optional[Image.Image]]:
        """Apply OCR filter defense.
        
        Args:
            image: Input image to check
            
        Returns:
            Tuple of (is_blocked, reason, processed_image)
            If blocked, processed_image is None
        """
        extracted_text = self.extract_text(image)
        is_harmful, keywords = self.contains_harmful_content(extracted_text)
        
        if is_harmful:
            return True, f"Blocked: detected harmful keywords: {keywords}", None
        
        return False, "Passed OCR filter", image


class GaussianBlurDefense:
    """Defense that applies Gaussian blur to disrupt adversarial patterns."""
    
    def __init__(self, config: Optional[InputDefenseConfig] = None):
        self.config = config or InputDefenseConfig()
    
    def defend(self, image: Image.Image) -> Tuple[bool, str, Image.Image]:
        """Apply Gaussian blur defense.
        
        Args:
            image: Input image to process
            
        Returns:
            Tuple of (is_blocked, reason, processed_image)
        """
        blurred = image.filter(ImageFilter.GaussianBlur(radius=self.config.blur_radius))
        return False, f"Applied Gaussian blur (radius={self.config.blur_radius})", blurred


class JPEGCompressionDefense:
    """Defense that applies JPEG compression to remove adversarial perturbations."""
    
    def __init__(self, config: Optional[InputDefenseConfig] = None):
        self.config = config or InputDefenseConfig()
    
    def defend(self, image: Image.Image) -> Tuple[bool, str, Image.Image]:
        """Apply JPEG compression defense.
        
        Args:
            image: Input image to process
            
        Returns:
            Tuple of (is_blocked, reason, processed_image)
        """
        import io
        
        # Save to JPEG buffer with specified quality
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.config.jpeg_quality)
        buffer.seek(0)
        
        # Reload from buffer
        compressed = Image.open(buffer).convert('RGB')
        
        return False, f"Applied JPEG compression (quality={self.config.jpeg_quality})", compressed


class ResizeDefense:
    """Defense that resizes image to disrupt pixel-level perturbations."""
    
    def __init__(self, config: Optional[InputDefenseConfig] = None):
        self.config = config or InputDefenseConfig()
    
    def defend(self, image: Image.Image) -> Tuple[bool, str, Image.Image]:
        """Apply resize defense.
        
        Args:
            image: Input image to process
            
        Returns:
            Tuple of (is_blocked, reason, processed_image)
        """
        original_size = image.size
        
        # Downscale
        small_size = (
            int(original_size[0] * self.config.resize_factor),
            int(original_size[1] * self.config.resize_factor)
        )
        small = image.resize(small_size, Image.BILINEAR)
        
        # Upscale back
        restored = small.resize(original_size, Image.BILINEAR)
        
        return False, f"Applied resize defense (factor={self.config.resize_factor})", restored


class CompositeInputDefense:
    """Combines multiple input-level defenses."""
    
    def __init__(self, config: Optional[InputDefenseConfig] = None):
        self.config = config or InputDefenseConfig()
        
        self.defenses = [
            ('ocr_filter', OCRFilterDefense(config)),
            ('gaussian_blur', GaussianBlurDefense(config)),
            ('jpeg_compression', JPEGCompressionDefense(config)),
        ]
    
    def defend(
        self,
        image: Image.Image,
        apply_all: bool = True
    ) -> Tuple[bool, List[str], Optional[Image.Image]]:
        """Apply composite defense.
        
        Args:
            image: Input image to process
            apply_all: If True, apply all defenses; if False, stop on first block
            
        Returns:
            Tuple of (is_blocked, reasons, processed_image)
        """
        reasons = []
        current_image = image
        
        for name, defense in self.defenses:
            is_blocked, reason, processed = defense.defend(current_image)
            reasons.append(f"{name}: {reason}")
            
            if is_blocked:
                if not apply_all:
                    return True, reasons, None
            else:
                current_image = processed
        
        # Check if any defense blocked
        is_blocked = any("Blocked" in r for r in reasons)
        
        return is_blocked, reasons, None if is_blocked else current_image
