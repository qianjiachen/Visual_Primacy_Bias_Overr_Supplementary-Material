"""Main experiment runner.

Orchestrates the complete experiment pipeline including:
- TAS attack generation
- Baseline comparisons
- Mechanistic analysis
- Defense evaluation
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import yaml
import torch
import numpy as np
from PIL import Image

from ..models.config import ExperimentConfig, AttackConfig
from ..models.results import ResultsCollector, AttackResult
from ..rendering.pipeline import RenderingPipeline
from ..scoring.gradient_scorer import GradientScorer
from ..evaluation.asr import ASRCalculator
from ..evaluation.ocr import OCREvasionEvaluator


@dataclass
class ExperimentState:
    """Tracks experiment progress for checkpointing."""
    current_phase: str = "init"
    completed_samples: int = 0
    total_samples: int = 0
    start_time: Optional[str] = None
    last_checkpoint: Optional[str] = None
    results_so_far: Dict = None
    
    def __post_init__(self):
        if self.results_so_far is None:
            self.results_so_far = {}


class ExperimentRunner:
    """Main experiment runner for Visual Primacy Bias reproduction."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "outputs",
        checkpoint_dir: str = "checkpoints"
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.results_collector = ResultsCollector()
        self.state = ExperimentState()
        
        # Will be initialized lazily
        self._models = None
        self._rendering_pipeline = None
        self._gradient_scorer = None
        self._asr_calculator = None
        self._ocr_evaluator = None
    
    @property
    def models(self) -> Dict:
        """Lazy load models."""
        if self._models is None:
            self._models = self._load_models()
        return self._models
    
    @property
    def rendering_pipeline(self) -> RenderingPipeline:
        """Lazy load rendering pipeline."""
        if self._rendering_pipeline is None:
            self._rendering_pipeline = RenderingPipeline(self.config.attack)
        return self._rendering_pipeline
    
    def _load_models(self) -> Dict:
        """Load all configured models."""
        models = {}
        
        for model_name in self.config.models:
            print(f"Loading model: {model_name}")
            try:
                if 'llava' in model_name.lower():
                    from ..models.llava import LLaVAModel
                    models[model_name] = LLaVAModel(model_name)
                elif 'qwen' in model_name.lower():
                    from ..models.qwen import QwenVLModel
                    models[model_name] = QwenVLModel(model_name)
                elif 'instructblip' in model_name.lower():
                    from ..models.instructblip import InstructBLIPModel
                    models[model_name] = InstructBLIPModel(model_name)
                elif 'gpt-4' in model_name.lower():
                    from ..models.gpt4v import GPT4VModel
                    models[model_name] = GPT4VModel()
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
        
        return models

    def save_checkpoint(self, phase: str = None):
        """Save experiment checkpoint."""
        if phase:
            self.state.current_phase = phase
        
        self.state.last_checkpoint = datetime.now().isoformat()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.state.current_phase}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str = None) -> bool:
        """Load experiment checkpoint."""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                return False
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            self.state = ExperimentState(**data)
            print(f"Loaded checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            return False
    
    def run_tas_attack(
        self,
        samples: List[Dict],
        model_name: str
    ) -> List[AttackResult]:
        """Run TAS attack on samples.
        
        Args:
            samples: List of sample dicts with 'query' and 'category'
            model_name: Target model name
            
        Returns:
            List of attack results
        """
        results = []
        model = self.models.get(model_name)
        
        if model is None:
            print(f"Model {model_name} not available")
            return results
        
        # Initialize gradient scorer for this model
        if hasattr(model, 'model'):
            scorer = GradientScorer(model.model, model.tokenizer)
        else:
            scorer = None
        
        for i, sample in enumerate(samples):
            query = sample.get('query', '')
            category = sample.get('category', 'unknown')
            
            print(f"Processing sample {i+1}/{len(samples)}: {query[:50]}...")
            
            try:
                # Generate candidate images
                candidates = self.rendering_pipeline.render_candidates(query)
                
                # Select best candidate using gradient scoring
                if scorer and candidates:
                    best_image, best_score = scorer.select_best_candidate(
                        candidates, query
                    )
                else:
                    best_image = candidates[0] if candidates else None
                    best_score = 0.0
                
                # Generate response
                if best_image:
                    response = model.generate(best_image, "Describe what you see.")
                else:
                    response = ""
                
                result = AttackResult(
                    sample_id=str(i),
                    model_name=model_name,
                    attack_type="TAS",
                    query=query,
                    response=response,
                    success=False,  # Will be evaluated later
                    category=category,
                    metadata={'score': best_score}
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                results.append(AttackResult(
                    sample_id=str(i),
                    model_name=model_name,
                    attack_type="TAS",
                    query=query,
                    response="",
                    success=False,
                    category=category,
                    metadata={'error': str(e)}
                ))
            
            # Checkpoint periodically
            if (i + 1) % 10 == 0:
                self.state.completed_samples = i + 1
                self.save_checkpoint("tas_attack")
        
        return results
    
    def run_baseline_attacks(
        self,
        samples: List[Dict],
        model_name: str
    ) -> Dict[str, List[AttackResult]]:
        """Run baseline attacks for comparison.
        
        Args:
            samples: List of sample dicts
            model_name: Target model name
            
        Returns:
            Dict mapping attack name to results
        """
        from ..baselines import (
            DirectInstructionAttack,
            FigStepAttack,
            GCGAttack,
            PGDAttack
        )
        
        results = {}
        model = self.models.get(model_name)
        
        if model is None:
            return results
        
        # Direct instruction baseline
        print("Running Direct Instruction baseline...")
        direct_attack = DirectInstructionAttack()
        direct_results = []
        
        for i, sample in enumerate(samples):
            image, prompt = direct_attack.attack(sample['query'])
            response = model.generate(image, prompt)
            direct_results.append(AttackResult(
                sample_id=str(i),
                model_name=model_name,
                attack_type="Direct",
                query=sample['query'],
                response=response,
                success=False,
                category=sample.get('category', 'unknown')
            ))
        
        results['direct'] = direct_results
        
        # FigStep baseline
        print("Running FigStep baseline...")
        figstep_attack = FigStepAttack()
        figstep_results = []
        
        for i, sample in enumerate(samples):
            image, prompt = figstep_attack.attack(sample['query'])
            response = model.generate(image, prompt)
            figstep_results.append(AttackResult(
                sample_id=str(i),
                model_name=model_name,
                attack_type="FigStep",
                query=sample['query'],
                response=response,
                success=False,
                category=sample.get('category', 'unknown')
            ))
        
        results['figstep'] = figstep_results
        
        return results
    
    def evaluate_results(self, results: List[AttackResult]) -> Dict[str, float]:
        """Evaluate attack results using ASR calculator."""
        if self._asr_calculator is None:
            self._asr_calculator = ASRCalculator()
        
        responses = [r.response for r in results]
        queries = [r.query for r in results]
        
        # Judge responses
        judgments = []
        for response, query in zip(responses, queries):
            judgment = self._asr_calculator.judge_with_gpt4(response, query)
            judgments.append(judgment)
        
        # Update results with judgments
        for result, judgment in zip(results, judgments):
            result.success = judgment
        
        # Compute ASR
        asr = self._asr_calculator.compute_asr(responses, queries)
        
        # Compute category-wise ASR
        category_asr = self._asr_calculator.compute_category_asr(
            responses, queries, [r.category for r in results]
        )
        
        return {
            'overall_asr': asr,
            'category_asr': category_asr,
            'num_samples': len(results),
            'num_successful': sum(1 for r in results if r.success)
        }
    
    def run_full_experiment(
        self,
        samples: List[Dict],
        resume: bool = True
    ) -> Dict[str, Any]:
        """Run the complete experiment pipeline.
        
        Args:
            samples: List of sample dicts
            resume: Whether to resume from checkpoint
            
        Returns:
            Complete experiment results
        """
        self.state.start_time = datetime.now().isoformat()
        self.state.total_samples = len(samples)
        
        # Try to resume
        if resume:
            self.load_checkpoint()
        
        all_results = {}
        
        # Run attacks on each model
        for model_name in self.config.models:
            print(f"\n{'='*50}")
            print(f"Running experiments on {model_name}")
            print(f"{'='*50}")
            
            # TAS attack
            tas_results = self.run_tas_attack(samples, model_name)
            tas_eval = self.evaluate_results(tas_results)
            
            # Baseline attacks
            baseline_results = self.run_baseline_attacks(samples, model_name)
            baseline_evals = {}
            for attack_name, results in baseline_results.items():
                baseline_evals[attack_name] = self.evaluate_results(results)
            
            all_results[model_name] = {
                'tas': tas_eval,
                'baselines': baseline_evals
            }
            
            self.save_checkpoint(f"completed_{model_name}")
        
        # Save final results
        self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        results_path = self.output_dir / "results" / "experiment_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_path}")
