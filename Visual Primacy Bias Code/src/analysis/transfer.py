"""Transfer attack analysis module.

Analyzes the transferability of TAS attacks across different models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import torch

from ..models.interface import MLLMInterface
from ..evaluation.asr import ASREvaluator


@dataclass
class TransferConfig:
    """Configuration for transfer analysis."""
    num_samples: int = 100
    seed: int = 42


class TransferAnalyzer:
    """Analyzes transfer attack success across models.
    
    Computes transfer ASR matrix showing how attacks generated
    on one model transfer to other models.
    """
    
    def __init__(
        self,
        models: Dict[str, MLLMInterface],
        asr_evaluator: Optional[ASREvaluator] = None,
        config: Optional[TransferConfig] = None
    ):
        """Initialize transfer analyzer.
        
        Args:
            models: Dict mapping model name to model interface
            asr_evaluator: ASR evaluator instance
            config: Transfer analysis configuration
        """
        self.models = models
        self.asr_evaluator = asr_evaluator or ASREvaluator()
        self.config = config or TransferConfig()
    
    def compute_transfer_matrix(
        self,
        attack_images: Dict[str, List[Image.Image]],
        prompts: List[str],
        harmful_behaviors: List[str],
        categories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute transfer ASR matrix.
        
        Args:
            attack_images: Dict mapping source model to list of attack images
            prompts: List of text prompts
            harmful_behaviors: List of target harmful behaviors
            categories: Optional list of harm categories
            
        Returns:
            Nested dict [source_model][target_model] -> transfer ASR
        """
        transfer_matrix = {}
        
        if categories is None:
            categories = ['general'] * len(prompts)
        
        for source_model in attack_images.keys():
            transfer_matrix[source_model] = {}
            images = attack_images[source_model]
            
            for target_name, target_model in self.models.items():
                # Generate responses on target model
                attacks = []
                for i, (img, prompt) in enumerate(zip(images, prompts)):
                    try:
                        response = target_model.generate(img, prompt)
                    except Exception as e:
                        print(f"Error generating on {target_name}: {e}")
                        response = ""
                    
                    attacks.append({
                        'harmful_request': harmful_behaviors[i] if i < len(harmful_behaviors) else prompt,
                        'response': response,
                        'category': categories[i] if i < len(categories) else 'general'
                    })
                
                # Compute ASR using batch evaluation
                summary, _ = self.asr_evaluator.batch_evaluate(attacks)
                transfer_matrix[source_model][target_name] = summary.overall_asr * 100  # Convert to percentage
        
        return transfer_matrix
    
    def analyze_universal_transferability(
        self,
        attack_images: List[Image.Image],
        prompts: List[str],
        harmful_behaviors: List[str],
        categories: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Analyze transferability of universal attacks.
        
        Args:
            attack_images: List of universal attack images
            prompts: List of text prompts
            harmful_behaviors: List of target harmful behaviors
            categories: Optional list of harm categories
            
        Returns:
            Dict mapping model name to ASR (percentage)
        """
        results = {}
        
        if categories is None:
            categories = ['general'] * len(prompts)
        
        for model_name, model in self.models.items():
            attacks = []
            for i, (img, prompt) in enumerate(zip(attack_images, prompts)):
                try:
                    response = model.generate(img, prompt)
                except Exception as e:
                    print(f"Error on {model_name}: {e}")
                    response = ""
                
                attacks.append({
                    'harmful_request': harmful_behaviors[i] if i < len(harmful_behaviors) else prompt,
                    'response': response,
                    'category': categories[i] if i < len(categories) else 'general'
                })
            
            summary, _ = self.asr_evaluator.batch_evaluate(attacks)
            results[model_name] = summary.overall_asr * 100  # Convert to percentage
        
        return results

    def compute_style_transfer_effectiveness(
        self,
        style_params: List[Dict],
        base_images: List[Image.Image],
        prompts: List[str],
        harmful_behaviors: List[str],
        categories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Analyze which style parameters transfer best.
        
        Args:
            style_params: List of style parameter dicts
            base_images: List of base images
            prompts: List of text prompts
            harmful_behaviors: List of target harmful behaviors
            categories: Optional list of harm categories
            
        Returns:
            Dict mapping style param to model ASRs (percentages)
        """
        results = {}
        
        if categories is None:
            categories = ['general'] * len(prompts)
        
        for params in style_params:
            param_key = str(params)
            results[param_key] = {}
            
            for model_name, model in self.models.items():
                attacks = []
                for i, (img, prompt) in enumerate(zip(base_images, prompts)):
                    try:
                        response = model.generate(img, prompt)
                    except Exception:
                        response = ""
                    
                    attacks.append({
                        'harmful_request': harmful_behaviors[i] if i < len(harmful_behaviors) else prompt,
                        'response': response,
                        'category': categories[i] if i < len(categories) else 'general'
                    })
                
                summary, _ = self.asr_evaluator.batch_evaluate(attacks)
                results[param_key][model_name] = summary.overall_asr * 100
        
        return results
    
    def find_universal_style(
        self,
        style_results: Dict[str, Dict[str, float]],
        min_asr_threshold: float = 0.5
    ) -> Optional[str]:
        """Find style parameters that work universally across models.
        
        Args:
            style_results: Results from compute_style_transfer_effectiveness
            min_asr_threshold: Minimum ASR required on all models
            
        Returns:
            Best universal style parameter key, or None
        """
        best_style = None
        best_min_asr = 0
        
        for style_key, model_asrs in style_results.items():
            min_asr = min(model_asrs.values())
            
            if min_asr >= min_asr_threshold and min_asr > best_min_asr:
                best_min_asr = min_asr
                best_style = style_key
        
        return best_style
    
    def compute_transfer_statistics(
        self,
        transfer_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute statistics from transfer matrix.
        
        Args:
            transfer_matrix: Transfer ASR matrix
            
        Returns:
            Dict of statistics
        """
        all_values = []
        diagonal_values = []
        off_diagonal_values = []
        
        models = list(transfer_matrix.keys())
        
        for source in models:
            for target in models:
                val = transfer_matrix[source].get(target, 0)
                all_values.append(val)
                
                if source == target:
                    diagonal_values.append(val)
                else:
                    off_diagonal_values.append(val)
        
        return {
            'mean_asr': np.mean(all_values),
            'std_asr': np.std(all_values),
            'mean_self_asr': np.mean(diagonal_values) if diagonal_values else 0,
            'mean_transfer_asr': np.mean(off_diagonal_values) if off_diagonal_values else 0,
            'transfer_gap': np.mean(diagonal_values) - np.mean(off_diagonal_values) if diagonal_values and off_diagonal_values else 0,
            'best_transfer_pair': self._find_best_transfer_pair(transfer_matrix, models),
            'worst_transfer_pair': self._find_worst_transfer_pair(transfer_matrix, models),
        }
    
    def _find_best_transfer_pair(
        self,
        matrix: Dict[str, Dict[str, float]],
        models: List[str]
    ) -> Tuple[str, str, float]:
        """Find the best transferring source-target pair."""
        best = (None, None, 0)
        
        for source in models:
            for target in models:
                if source != target:
                    val = matrix[source].get(target, 0)
                    if val > best[2]:
                        best = (source, target, val)
        
        return best
    
    def _find_worst_transfer_pair(
        self,
        matrix: Dict[str, Dict[str, float]],
        models: List[str]
    ) -> Tuple[str, str, float]:
        """Find the worst transferring source-target pair."""
        worst = (None, None, 100)
        
        for source in models:
            for target in models:
                if source != target:
                    val = matrix[source].get(target, 100)
                    if val < worst[2]:
                        worst = (source, target, val)
        
        return worst
