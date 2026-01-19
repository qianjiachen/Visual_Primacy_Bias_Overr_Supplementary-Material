"""TAS Image Rendering Pipeline.

This module implements the Typographic Attention Steering (TAS) rendering pipeline
for generating adversarial images with stylized typography.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random
import math
from pathlib import Path
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter

from src.models.config import DistortionParams, DistortionConfig, RenderConfig
from .background import BackgroundGenerator


# Default fonts available on most systems
DEFAULT_FONTS = [
    # System fonts (fallbacks)
    "arial.ttf",
    "times.ttf",
    "verdana.ttf",
    "georgia.ttf",
    "comic.ttf",
    "impact.ttf",
    "trebuc.ttf",
    "tahoma.ttf",
    "calibri.ttf",
    "consola.ttf",
    # Additional common fonts
    "Arial",
    "Times New Roman",
    "Verdana",
    "Georgia",
    "Comic Sans MS",
]


class FontManager:
    """Manages font loading and caching."""
    
    def __init__(self, custom_font_dir: Optional[str] = None):
        self.custom_font_dir = Path(custom_font_dir) if custom_font_dir else None
        self._font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._available_fonts: List[str] = []
        self._discover_fonts()
    
    def _discover_fonts(self) -> None:
        """Discover available fonts on the system."""
        # Try to load each default font
        for font_name in DEFAULT_FONTS:
            try:
                font = ImageFont.truetype(font_name, 32)
                self._available_fonts.append(font_name)
            except (OSError, IOError):
                pass
        
        # Check custom font directory
        if self.custom_font_dir and self.custom_font_dir.exists():
            for font_file in self.custom_font_dir.glob("*.ttf"):
                self._available_fonts.append(str(font_file))
            for font_file in self.custom_font_dir.glob("*.otf"):
                self._available_fonts.append(str(font_file))
        
        # Fallback to default if no fonts found
        if not self._available_fonts:
            self._available_fonts = ["arial.ttf", "Arial"]
    
    def get_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Get a font with caching."""
        cache_key = (font_name, size)
        
        if cache_key not in self._font_cache:
            try:
                font = ImageFont.truetype(font_name, size)
            except (OSError, IOError):
                # Fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            self._font_cache[cache_key] = font
        
        return self._font_cache[cache_key]
    
    @property
    def available_fonts(self) -> List[str]:
        """Get list of available fonts."""
        return self._available_fonts.copy()


class RenderingPipeline:
    """TAS Image Rendering Pipeline.
    
    Renders harmful instructions into stylized typography images
    with various distortions to evade OCR while remaining readable
    by Vision Transformers.
    """
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """Initialize the rendering pipeline.
        
        Args:
            config: Rendering configuration. Uses defaults if None.
        """
        self.config = config or RenderConfig()
        self.font_manager = FontManager()
        self.background_generator = BackgroundGenerator()
        
        # Ensure we have at least 15 fonts
        self._fonts = self._setup_fonts()
    
    def _setup_fonts(self) -> List[str]:
        """Setup font list, ensuring at least 15 fonts."""
        fonts = self.font_manager.available_fonts
        
        # If we have fewer than 15, duplicate with variations
        while len(fonts) < 15:
            fonts = fonts + fonts[:15 - len(fonts)]
        
        return fonts[:15]
    
    @property
    def fonts(self) -> List[str]:
        """Get list of available fonts."""
        return self._fonts.copy()
    
    def render(
        self,
        text: str,
        font: Optional[str] = None,
        distortion: Optional[DistortionParams] = None,
        size: Optional[Tuple[int, int]] = None,
        font_size: int = 48,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        background_type: Optional[str] = None,
    ) -> np.ndarray:
        """Render text into an image with distortions.
        
        Args:
            text: Text to render
            font: Font name (uses random if None)
            distortion: Distortion parameters (uses random if None)
            size: Image size (width, height)
            font_size: Font size in pixels
            text_color: RGB color tuple for text
            background_type: Type of background (perlin, gradient, solid)
        
        Returns:
            Rendered image as numpy array (H, W, 3) in RGB format
        """
        size = size or self.config.image_size
        font = font or random.choice(self._fonts)
        distortion = distortion or self.config.distortion.sample()
        background_type = background_type or self.config.background_type
        
        # Validate and clamp distortion parameters
        if not distortion.validate():
            distortion = distortion.clamp()
        
        # Generate background
        background = self.background_generator.generate(
            size=size,
            bg_type=background_type,
            octaves=random.randint(*self.config.perlin_octaves_range),
            persistence=random.uniform(*self.config.perlin_persistence_range),
        )
        
        # Create image from background
        image = Image.fromarray(background)
        
        # Render text onto image
        image = self._render_text(image, text, font, font_size, text_color)
        
        # Apply distortions
        image_array = np.array(image)
        image_array = self._apply_distortions(image_array, distortion)
        
        return image_array
    
    def _render_text(
        self,
        image: Image.Image,
        text: str,
        font_name: str,
        font_size: int,
        text_color: Tuple[int, int, int],
    ) -> Image.Image:
        """Render text onto image."""
        draw = ImageDraw.Draw(image)
        font = self.font_manager.get_font(font_name, font_size)
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2
        
        # Handle multi-line text
        lines = text.split('\n')
        if len(lines) > 1:
            total_height = text_height * len(lines)
            y = (image.height - total_height) // 2
            
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_x = (image.width - line_width) // 2
                line_y = y + i * text_height
                draw.text((line_x, line_y), line, font=font, fill=text_color)
        else:
            draw.text((x, y), text, font=font, fill=text_color)
        
        return image
    
    def _apply_distortions(
        self,
        image: np.ndarray,
        params: DistortionParams,
    ) -> np.ndarray:
        """Apply all distortions to image."""
        # Apply elastic deformation
        image = self._elastic_transform(
            image,
            alpha=params.elastic_alpha,
            sigma=params.elastic_sigma,
        )
        
        # Apply wave distortion
        image = self._wave_distortion(
            image,
            amplitude=params.wave_amplitude,
            frequency=params.wave_frequency,
        )
        
        # Apply rotation
        if abs(params.rotation) > 0.1:
            image = self._rotate(image, params.rotation)
        
        # Apply perspective transform
        if params.perspective_displacement > 0.01:
            image = self._perspective_transform(
                image,
                displacement=params.perspective_displacement,
            )
        
        return image
    
    def _elastic_transform(
        self,
        image: np.ndarray,
        alpha: float,
        sigma: float,
    ) -> np.ndarray:
        """Apply elastic deformation to image.
        
        Based on: Simard et al., "Best Practices for Convolutional Neural Networks"
        """
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma,
            mode="constant",
            cval=0
        ) * alpha
        
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            sigma,
            mode="constant",
            cval=0
        ) * alpha
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        # Apply displacement
        indices_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        indices_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
        
        # Remap image
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = map_coordinates(
                    image[:, :, c],
                    [indices_y, indices_x],
                    order=1,
                    mode='reflect'
                )
            return result
        else:
            return map_coordinates(
                image,
                [indices_y, indices_x],
                order=1,
                mode='reflect'
            )
    
    def _wave_distortion(
        self,
        image: np.ndarray,
        amplitude: float,
        frequency: float,
    ) -> np.ndarray:
        """Apply sinusoidal wave distortion."""
        rows, cols = image.shape[:2]
        
        # Create output image
        result = np.zeros_like(image)
        
        for i in range(rows):
            # Calculate horizontal shift
            shift = int(amplitude * np.sin(2 * np.pi * i * frequency))
            
            # Shift row
            if shift > 0:
                result[i, shift:] = image[i, :-shift] if shift < cols else 0
                result[i, :shift] = image[i, 0]  # Fill edge
            elif shift < 0:
                result[i, :shift] = image[i, -shift:] if -shift < cols else 0
                result[i, shift:] = image[i, -1]  # Fill edge
            else:
                result[i] = image[i]
        
        return result
    
    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle degrees."""
        rows, cols = image.shape[:2]
        center = (cols // 2, rows // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            M,
            (cols, rows),
            borderMode=cv2.BORDER_REFLECT,
        )
        
        return rotated
    
    def _perspective_transform(
        self,
        image: np.ndarray,
        displacement: float,
    ) -> np.ndarray:
        """Apply random perspective transform."""
        rows, cols = image.shape[:2]
        
        # Calculate max displacement in pixels
        max_disp = int(min(rows, cols) * displacement)
        
        # Source points (corners)
        src_pts = np.float32([
            [0, 0],
            [cols - 1, 0],
            [cols - 1, rows - 1],
            [0, rows - 1]
        ])
        
        # Destination points (with random displacement)
        dst_pts = np.float32([
            [random.randint(0, max_disp), random.randint(0, max_disp)],
            [cols - 1 - random.randint(0, max_disp), random.randint(0, max_disp)],
            [cols - 1 - random.randint(0, max_disp), rows - 1 - random.randint(0, max_disp)],
            [random.randint(0, max_disp), rows - 1 - random.randint(0, max_disp)]
        ])
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transform
        warped = cv2.warpPerspective(
            image,
            M,
            (cols, rows),
            borderMode=cv2.BORDER_REFLECT,
        )
        
        return warped
    
    def render_candidates(
        self,
        text: str,
        num_candidates: int = 120,
        font_size: int = 48,
    ) -> List[Tuple[np.ndarray, str, DistortionParams]]:
        """Generate multiple candidate images with different styles.
        
        Args:
            text: Text to render
            num_candidates: Number of candidates to generate
            font_size: Font size in pixels
        
        Returns:
            List of (image, font_name, distortion_params) tuples
        """
        candidates = []
        
        # Calculate how many variations per font
        num_fonts = len(self._fonts)
        variations_per_font = max(1, num_candidates // num_fonts)
        
        for font in self._fonts:
            for _ in range(variations_per_font):
                if len(candidates) >= num_candidates:
                    break
                
                distortion = self.config.distortion.sample()
                image = self.render(
                    text=text,
                    font=font,
                    distortion=distortion,
                    font_size=font_size,
                )
                candidates.append((image, font, distortion))
        
        # Fill remaining with random combinations
        while len(candidates) < num_candidates:
            font = random.choice(self._fonts)
            distortion = self.config.distortion.sample()
            image = self.render(
                text=text,
                font=font,
                distortion=distortion,
                font_size=font_size,
            )
            candidates.append((image, font, distortion))
        
        return candidates[:num_candidates]
    
    def save_image(
        self,
        image: np.ndarray,
        path: str,
        format: str = "PNG",
    ) -> None:
        """Save image to file.
        
        Args:
            image: Image array (H, W, 3) in RGB format
            path: Output file path
            format: Image format (PNG, JPEG)
        """
        img = Image.fromarray(image.astype(np.uint8))
        img.save(path, format=format)
