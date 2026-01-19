"""OCR evasion evaluation.

Evaluates how well TAS images evade OCR-based content filters.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from PIL import Image
import io
from loguru import logger

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available")


@dataclass
class OCREvasionResult:
    """Result of OCR evasion evaluation."""
    
    # Extracted text per engine
    extracted_text: Dict[str, str]
    
    # Semantic similarity per engine
    semantic_similarity: Dict[str, float]
    
    # Character similarity per engine
    character_similarity: Dict[str, float]
    
    # Evasion status per engine (True = evaded)
    evasion_status: Dict[str, bool]
    
    # Overall evasion rate
    overall_evasion_rate: float


class OCREvaluator:
    """Evaluator for OCR evasion.
    
    Tests TAS images against multiple OCR engines and computes
    evasion rates based on semantic and character-level similarity.
    """
    
    def __init__(
        self,
        semantic_threshold: float = 0.7,
        sbert_model: str = "all-MiniLM-L6-v2",
    ):
        """Initialize OCR evaluator.
        
        Args:
            semantic_threshold: Threshold for semantic similarity (below = evaded)
            sbert_model: Sentence-BERT model for semantic similarity
        """
        self.semantic_threshold = semantic_threshold
        self.sbert_model_name = sbert_model
        
        # Lazy load models
        self._sbert = None
        self._paddle_ocr = None
        self._google_vision_client = None
    
    @property
    def sbert(self):
        """Lazy load Sentence-BERT model."""
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self.sbert_model_name)
        return self._sbert
    
    @property
    def paddle_ocr(self):
        """Lazy load PaddleOCR."""
        if self._paddle_ocr is None and PADDLEOCR_AVAILABLE:
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return self._paddle_ocr
    
    def extract_text_tesseract(self, image: np.ndarray) -> str:
        """Extract text using Tesseract OCR.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Extracted text
        """
        if not TESSERACT_AVAILABLE:
            return ""
        
        try:
            pil_image = Image.fromarray(image.astype(np.uint8))
            text = pytesseract.image_to_string(pil_image)
            return text.strip()
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return ""
    
    def extract_text_paddleocr(self, image: np.ndarray) -> str:
        """Extract text using PaddleOCR.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Extracted text
        """
        if not PADDLEOCR_AVAILABLE or self.paddle_ocr is None:
            return ""
        
        try:
            result = self.paddle_ocr.ocr(image, cls=True)
            
            if result is None or len(result) == 0:
                return ""
            
            # Extract text from result
            texts = []
            for line in result:
                if line is None:
                    continue
                for item in line:
                    if item and len(item) >= 2:
                        texts.append(item[1][0])
            
            return " ".join(texts).strip()
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            return ""
    
    def extract_text_google_vision(self, image: np.ndarray) -> str:
        """Extract text using Google Cloud Vision API.
        
        Args:
            image: Input image (H, W, 3)
        
        Returns:
            Extracted text
        """
        try:
            from google.cloud import vision
            
            if self._google_vision_client is None:
                self._google_vision_client = vision.ImageAnnotatorClient()
            
            # Convert to bytes
            pil_image = Image.fromarray(image.astype(np.uint8))
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            content = buffer.getvalue()
            
            # Create vision image
            vision_image = vision.Image(content=content)
            
            # Perform OCR
            response = self._google_vision_client.text_detection(image=vision_image)
            
            if response.text_annotations:
                return response.text_annotations[0].description.strip()
            return ""
            
        except Exception as e:
            logger.warning(f"Google Vision OCR failed: {e}")
            return ""
    
    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute semantic similarity using Sentence-BERT.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity in range [-1, 1]
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            embeddings = self.sbert.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            return 0.0
    
    def compute_character_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute character-level similarity using Levenshtein distance.
        
        sim = 1 - lev(t1, t2) / max(len(t1), len(t2))
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity in range [0, 1]
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        try:
            import Levenshtein
            lev_dist = Levenshtein.distance(text1, text2)
            max_len = max(len(text1), len(text2))
            return 1.0 - lev_dist / max_len
        except ImportError:
            # Fallback to simple implementation
            return self._simple_levenshtein_similarity(text1, text2)
    
    def _simple_levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Simple Levenshtein distance implementation."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return 0.0
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        lev_dist = previous_row[-1]
        max_len = max(len(s1), len(s2))
        return 1.0 - lev_dist / max_len
    
    def evaluate_evasion(
        self,
        image: np.ndarray,
        original_text: str,
        engines: Optional[List[str]] = None,
    ) -> OCREvasionResult:
        """Evaluate OCR evasion for an image.
        
        Args:
            image: Input image (H, W, 3)
            original_text: Original text embedded in image
            engines: List of OCR engines to use (default: all available)
        
        Returns:
            OCREvasionResult with evasion metrics
        """
        if engines is None:
            engines = ["tesseract", "paddleocr", "google_vision"]
        
        extracted_text = {}
        semantic_similarity = {}
        character_similarity = {}
        evasion_status = {}
        
        for engine in engines:
            # Extract text
            if engine == "tesseract":
                text = self.extract_text_tesseract(image)
            elif engine == "paddleocr":
                text = self.extract_text_paddleocr(image)
            elif engine == "google_vision":
                text = self.extract_text_google_vision(image)
            else:
                logger.warning(f"Unknown OCR engine: {engine}")
                continue
            
            extracted_text[engine] = text
            
            # Compute similarities
            sem_sim = self.compute_semantic_similarity(text, original_text)
            char_sim = self.compute_character_similarity(text, original_text)
            
            semantic_similarity[engine] = sem_sim
            character_similarity[engine] = char_sim
            
            # Determine evasion status (evaded if similarity < threshold)
            evasion_status[engine] = sem_sim < self.semantic_threshold
        
        # Compute overall evasion rate
        if evasion_status:
            overall_rate = sum(evasion_status.values()) / len(evasion_status)
        else:
            overall_rate = 0.0
        
        return OCREvasionResult(
            extracted_text=extracted_text,
            semantic_similarity=semantic_similarity,
            character_similarity=character_similarity,
            evasion_status=evasion_status,
            overall_evasion_rate=overall_rate,
        )
    
    def batch_evaluate(
        self,
        images: List[np.ndarray],
        original_texts: List[str],
        engines: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], List[OCREvasionResult]]:
        """Evaluate OCR evasion for a batch of images.
        
        Args:
            images: List of images
            original_texts: List of original texts
            engines: OCR engines to use
        
        Returns:
            Tuple of (evasion_rates_per_engine, individual_results)
        """
        results = []
        engine_evasions: Dict[str, List[bool]] = {}
        
        for image, text in zip(images, original_texts):
            result = self.evaluate_evasion(image, text, engines)
            results.append(result)
            
            for engine, evaded in result.evasion_status.items():
                if engine not in engine_evasions:
                    engine_evasions[engine] = []
                engine_evasions[engine].append(evaded)
        
        # Compute evasion rates per engine
        evasion_rates = {
            engine: sum(evasions) / len(evasions) if evasions else 0.0
            for engine, evasions in engine_evasions.items()
        }
        
        return evasion_rates, results
