"""Background generation for TAS images.

Implements procedural noise backgrounds (Perlin noise, gradients)
to create visually complex backgrounds that mask embedded text.
"""

import numpy as np
from typing import Tuple, Optional
import random
import math


class PerlinNoise:
    """Perlin noise generator for procedural backgrounds."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize Perlin noise generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate permutation table
        self.perm = np.arange(256, dtype=np.int32)
        np.random.shuffle(self.perm)
        self.perm = np.stack([self.perm, self.perm]).flatten()
        
        # Gradient vectors
        self.gradients = np.array([
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ], dtype=np.float32)
    
    def _fade(self, t: np.ndarray) -> np.ndarray:
        """Fade function for smooth interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Linear interpolation."""
        return a + t * (b - a)
    
    def _gradient(self, h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate gradient contribution."""
        g = self.gradients[h % 8]
        return g[..., 0] * x + g[..., 1] * y
    
    def noise2d(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate 2D Perlin noise.
        
        Args:
            x: X coordinates
            y: Y coordinates
        
        Returns:
            Noise values in range [-1, 1]
        """
        # Grid cell coordinates
        xi = x.astype(np.int32) & 255
        yi = y.astype(np.int32) & 255
        
        # Relative position within cell
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        
        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)
        
        # Hash coordinates of corners
        aa = self.perm[self.perm[xi] + yi]
        ab = self.perm[self.perm[xi] + yi + 1]
        ba = self.perm[self.perm[xi + 1] + yi]
        bb = self.perm[self.perm[xi + 1] + yi + 1]
        
        # Gradient contributions
        x1 = self._lerp(
            self._gradient(aa, xf, yf),
            self._gradient(ba, xf - 1, yf),
            u
        )
        x2 = self._lerp(
            self._gradient(ab, xf, yf - 1),
            self._gradient(bb, xf - 1, yf - 1),
            u
        )
        
        return self._lerp(x1, x2, v)
    
    def octave_noise(
        self,
        x: np.ndarray,
        y: np.ndarray,
        octaves: int = 6,
        persistence: float = 0.5,
    ) -> np.ndarray:
        """Generate multi-octave Perlin noise.
        
        Args:
            x: X coordinates
            y: Y coordinates
            octaves: Number of octaves
            persistence: Amplitude decay per octave
        
        Returns:
            Noise values normalized to [0, 1]
        """
        total = np.zeros_like(x)
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            total += self.noise2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2
        
        # Normalize to [0, 1]
        return (total / max_value + 1) / 2


class BackgroundGenerator:
    """Generates procedural backgrounds for TAS images."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize background generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.perlin = PerlinNoise(seed)
    
    def generate(
        self,
        size: Tuple[int, int],
        bg_type: str = "perlin",
        **kwargs,
    ) -> np.ndarray:
        """Generate a background image.
        
        Args:
            size: Image size (width, height)
            bg_type: Background type (perlin, gradient, solid)
            **kwargs: Additional parameters for specific background types
        
        Returns:
            Background image as numpy array (H, W, 3) in RGB format
        """
        width, height = size
        
        if bg_type == "perlin":
            return self.generate_perlin_noise(
                size=size,
                octaves=kwargs.get("octaves", 6),
                persistence=kwargs.get("persistence", 0.5),
                scale=kwargs.get("scale", 0.01),
            )
        elif bg_type == "gradient":
            return self.generate_gradient(
                size=size,
                direction=kwargs.get("direction", "diagonal"),
                colors=kwargs.get("colors", None),
            )
        elif bg_type == "solid":
            return self.generate_solid(
                size=size,
                color=kwargs.get("color", None),
            )
        else:
            # Default to perlin
            return self.generate_perlin_noise(size=size)
    
    def generate_perlin_noise(
        self,
        size: Tuple[int, int],
        octaves: int = 6,
        persistence: float = 0.5,
        scale: float = 0.01,
        color_mode: str = "grayscale",
    ) -> np.ndarray:
        """Generate Perlin noise background.
        
        Args:
            size: Image size (width, height)
            octaves: Number of noise octaves
            persistence: Amplitude decay per octave
            scale: Noise scale (smaller = larger features)
            color_mode: Color mode (grayscale, colored)
        
        Returns:
            Background image as numpy array (H, W, 3)
        """
        width, height = size
        
        # Create coordinate grids
        x = np.linspace(0, width * scale, width)
        y = np.linspace(0, height * scale, height)
        xx, yy = np.meshgrid(x, y)
        
        if color_mode == "grayscale":
            # Generate single channel noise
            noise = self.perlin.octave_noise(xx, yy, octaves, persistence)
            
            # Convert to RGB grayscale
            noise_uint8 = (noise * 255).astype(np.uint8)
            background = np.stack([noise_uint8, noise_uint8, noise_uint8], axis=-1)
        
        else:  # colored
            # Generate separate noise for each channel
            backgrounds = []
            for i in range(3):
                # Offset coordinates for each channel
                offset = i * 100
                noise = self.perlin.octave_noise(
                    xx + offset,
                    yy + offset,
                    octaves,
                    persistence
                )
                backgrounds.append((noise * 255).astype(np.uint8))
            
            background = np.stack(backgrounds, axis=-1)
        
        return background
    
    def generate_gradient(
        self,
        size: Tuple[int, int],
        direction: str = "diagonal",
        colors: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Generate gradient background.
        
        Args:
            size: Image size (width, height)
            direction: Gradient direction (horizontal, vertical, diagonal, radial)
            colors: Start and end colors as RGB tuples
        
        Returns:
            Background image as numpy array (H, W, 3)
        """
        width, height = size
        
        # Default colors
        if colors is None:
            colors = (
                (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255)),
                (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180)),
            )
        
        color1, color2 = colors
        
        # Create gradient based on direction
        if direction == "horizontal":
            t = np.linspace(0, 1, width)
            t = np.tile(t, (height, 1))
        
        elif direction == "vertical":
            t = np.linspace(0, 1, height)
            t = np.tile(t.reshape(-1, 1), (1, width))
        
        elif direction == "diagonal":
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            xx, yy = np.meshgrid(x, y)
            t = (xx + yy) / 2
        
        elif direction == "radial":
            x = np.linspace(-1, 1, width)
            y = np.linspace(-1, 1, height)
            xx, yy = np.meshgrid(x, y)
            t = np.sqrt(xx**2 + yy**2) / np.sqrt(2)
            t = np.clip(t, 0, 1)
        
        else:
            t = np.zeros((height, width))
        
        # Interpolate colors
        background = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            background[:, :, c] = (
                color1[c] * (1 - t) + color2[c] * t
            ).astype(np.uint8)
        
        return background
    
    def generate_solid(
        self,
        size: Tuple[int, int],
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Generate solid color background.
        
        Args:
            size: Image size (width, height)
            color: RGB color tuple
        
        Returns:
            Background image as numpy array (H, W, 3)
        """
        width, height = size
        
        if color is None:
            # Random light color
            color = (
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255),
            )
        
        background = np.full((height, width, 3), color, dtype=np.uint8)
        return background
    
    def generate_noisy_solid(
        self,
        size: Tuple[int, int],
        base_color: Optional[Tuple[int, int, int]] = None,
        noise_level: float = 20.0,
    ) -> np.ndarray:
        """Generate solid color background with noise.
        
        Args:
            size: Image size (width, height)
            base_color: Base RGB color
            noise_level: Standard deviation of noise
        
        Returns:
            Background image as numpy array (H, W, 3)
        """
        width, height = size
        
        if base_color is None:
            base_color = (220, 220, 220)
        
        # Create base
        background = np.full((height, width, 3), base_color, dtype=np.float32)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, (height, width, 3))
        background = background + noise
        
        # Clip to valid range
        background = np.clip(background, 0, 255).astype(np.uint8)
        
        return background
    
    def generate_texture(
        self,
        size: Tuple[int, int],
        texture_type: str = "paper",
    ) -> np.ndarray:
        """Generate textured background.
        
        Args:
            size: Image size (width, height)
            texture_type: Type of texture (paper, canvas, fabric)
        
        Returns:
            Background image as numpy array (H, W, 3)
        """
        width, height = size
        
        if texture_type == "paper":
            # Paper-like texture with fine noise
            base = self.generate_perlin_noise(
                size=size,
                octaves=8,
                persistence=0.4,
                scale=0.02,
            )
            # Add fine grain
            grain = np.random.normal(0, 5, (height, width, 3))
            background = np.clip(base.astype(np.float32) + grain, 0, 255)
            return background.astype(np.uint8)
        
        elif texture_type == "canvas":
            # Canvas-like texture with horizontal/vertical lines
            base = self.generate_solid(size, (240, 235, 220))
            
            # Add horizontal lines
            for y in range(0, height, 3):
                base[y, :, :] = np.clip(
                    base[y, :, :].astype(np.int32) - 10,
                    0, 255
                ).astype(np.uint8)
            
            # Add vertical lines
            for x in range(0, width, 3):
                base[:, x, :] = np.clip(
                    base[:, x, :].astype(np.int32) - 10,
                    0, 255
                ).astype(np.uint8)
            
            return base
        
        else:
            return self.generate_perlin_noise(size=size)
