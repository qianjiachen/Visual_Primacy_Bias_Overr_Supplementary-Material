"""Result data models for experiments."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
from datetime import datetime
import torch


@dataclass
class AttackResult:
    """Result of a single attack attempt."""
    
    # Input info
    image_path: str
    harmful_instruction: str
    category: str
    
    # Model response
    model_name: str
    model_response: str
    
    # Evaluation
    success: bool
    judge_explanation: Optional[str] = None
    
    # Mechanistic metrics (optional)
    attention_ratio: Optional[float] = None
    visual_dominant_heads: Optional[List[Tuple[int, int]]] = None
    
    # OCR evasion (optional)
    ocr_evasion: Optional[Dict[str, bool]] = None
    ocr_extracted_text: Optional[Dict[str, str]] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    attack_method: str = "TAS"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "harmful_instruction": self.harmful_instruction,
            "category": self.category,
            "model_name": self.model_name,
            "model_response": self.model_response,
            "success": self.success,
            "judge_explanation": self.judge_explanation,
            "attention_ratio": self.attention_ratio,
            "visual_dominant_heads": self.visual_dominant_heads,
            "ocr_evasion": self.ocr_evasion,
            "ocr_extracted_text": self.ocr_extracted_text,
            "timestamp": self.timestamp,
            "attack_method": self.attack_method,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackResult":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MechanisticAnalysisResult:
    """Result of mechanistic analysis on a model."""
    
    model_name: str
    
    # Attention analysis
    attention_ratios_by_layer: Dict[int, float] = field(default_factory=dict)
    attention_ratios_by_head: Dict[Tuple[int, int], float] = field(default_factory=dict)
    visual_dominant_heads: List[Tuple[int, int]] = field(default_factory=list)
    
    # Hidden state norms
    visual_token_norms: List[float] = field(default_factory=list)
    system_token_norms: List[float] = field(default_factory=list)
    
    # Attribution scores
    ig_attribution_visual: Optional[float] = None
    ig_attribution_system: Optional[float] = None
    
    # QKV ablation results
    qkv_ablation_results: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    # Logit lens contributions
    logit_lens_visual: List[float] = field(default_factory=list)
    logit_lens_system: List[float] = field(default_factory=list)
    
    # Cross-method correlation
    method_correlations: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    input_type: str = "successful_attack"  # successful_attack, failed_attack, benign
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (handles tuple keys)."""
        return {
            "model_name": self.model_name,
            "attention_ratios_by_layer": self.attention_ratios_by_layer,
            "attention_ratios_by_head": {
                f"{l},{h}": v for (l, h), v in self.attention_ratios_by_head.items()
            },
            "visual_dominant_heads": self.visual_dominant_heads,
            "visual_token_norms": self.visual_token_norms,
            "system_token_norms": self.system_token_norms,
            "ig_attribution_visual": self.ig_attribution_visual,
            "ig_attribution_system": self.ig_attribution_system,
            "qkv_ablation_results": {
                f"{l},{h}": v for (l, h), v in self.qkv_ablation_results.items()
            },
            "logit_lens_visual": self.logit_lens_visual,
            "logit_lens_system": self.logit_lens_system,
            "method_correlations": self.method_correlations,
            "input_type": self.input_type,
            "timestamp": self.timestamp,
        }


@dataclass
class CausalInterventionResult:
    """Result of causal intervention experiment."""
    
    model_name: str
    intervention_type: str  # attention_rebalancing, head_ablation
    
    # Intervention parameters
    gamma: Optional[float] = None  # For attention rebalancing
    ablated_heads: Optional[List[Tuple[int, int]]] = None
    
    # Results
    baseline_asr: float = 0.0
    intervention_asr: float = 0.0
    asr_delta: float = 0.0
    
    # Benign performance impact
    baseline_vqa_accuracy: Optional[float] = None
    intervention_vqa_accuracy: Optional[float] = None
    
    # Sample size
    num_samples: int = 0
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "intervention_type": self.intervention_type,
            "gamma": self.gamma,
            "ablated_heads": self.ablated_heads,
            "baseline_asr": self.baseline_asr,
            "intervention_asr": self.intervention_asr,
            "asr_delta": self.asr_delta,
            "baseline_vqa_accuracy": self.baseline_vqa_accuracy,
            "intervention_vqa_accuracy": self.intervention_vqa_accuracy,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
        }


@dataclass
class TransferResult:
    """Result of cross-model transfer experiment."""
    
    source_model: str
    target_model: str
    
    # Transfer ASR
    source_asr: float = 0.0
    transfer_asr: float = 0.0
    
    # Sample info
    num_samples: int = 0
    successful_transfers: int = 0
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_model": self.source_model,
            "target_model": self.target_model,
            "source_asr": self.source_asr,
            "transfer_asr": self.transfer_asr,
            "num_samples": self.num_samples,
            "successful_transfers": self.successful_transfers,
            "timestamp": self.timestamp,
        }


@dataclass
class DefenseEvalResult:
    """Result of defense evaluation."""
    
    defense_name: str
    model_name: str
    
    # ASR comparison
    baseline_asr: float = 0.0
    defended_asr: float = 0.0
    asr_reduction: float = 0.0
    
    # Defense parameters
    defense_params: Dict[str, Any] = field(default_factory=dict)
    
    # Sample info
    num_samples: int = 0
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "defense_name": self.defense_name,
            "model_name": self.model_name,
            "baseline_asr": self.baseline_asr,
            "defended_asr": self.defended_asr,
            "asr_reduction": self.asr_reduction,
            "defense_params": self.defense_params,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
        }


@dataclass
class ExperimentSummary:
    """Summary of a complete experiment run."""
    
    experiment_name: str
    config_path: str
    
    # Overall results
    total_attacks: int = 0
    successful_attacks: int = 0
    overall_asr: float = 0.0
    
    # Per-model results
    model_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-category results
    category_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # OCR evasion summary
    ocr_evasion_rates: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    total_duration_seconds: float = 0.0
    
    # Compute info
    gpu_memory_peak_gb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "config_path": self.config_path,
            "total_attacks": self.total_attacks,
            "successful_attacks": self.successful_attacks,
            "overall_asr": self.overall_asr,
            "model_results": self.model_results,
            "category_results": self.category_results,
            "ocr_evasion_rates": self.ocr_evasion_rates,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": self.total_duration_seconds,
            "gpu_memory_peak_gb": self.gpu_memory_peak_gb,
        }
    
    def save(self, path: str) -> None:
        """Save summary to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentSummary":
        """Load summary from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ResultsCollector:
    """Utility class for collecting and aggregating results."""
    
    def __init__(self):
        self.attack_results: List[AttackResult] = []
        self.mechanistic_results: List[MechanisticAnalysisResult] = []
        self.intervention_results: List[CausalInterventionResult] = []
        self.transfer_results: List[TransferResult] = []
        self.defense_results: List[DefenseEvalResult] = []
    
    def add_attack_result(self, result: AttackResult) -> None:
        """Add an attack result."""
        self.attack_results.append(result)
    
    def compute_asr(self, model_name: Optional[str] = None, 
                    category: Optional[str] = None) -> float:
        """Compute ASR with optional filtering."""
        results = self.attack_results
        
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        if category:
            results = [r for r in results if r.category == category]
        
        if not results:
            return 0.0
        
        return sum(1 for r in results if r.success) / len(results)
    
    def compute_category_asr(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """Compute ASR by category."""
        results = self.attack_results
        if model_name:
            results = [r for r in results if r.model_name == model_name]
        
        categories: Dict[str, Dict[str, int]] = {}
        for r in results:
            if r.category not in categories:
                categories[r.category] = {"success": 0, "total": 0}
            categories[r.category]["total"] += 1
            if r.success:
                categories[r.category]["success"] += 1
        
        return {
            cat: data["success"] / data["total"] if data["total"] > 0 else 0.0
            for cat, data in categories.items()
        }
    
    def save_all(self, output_dir: str) -> None:
        """Save all results to output directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save attack results
        with open(output_dir / "attack_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.attack_results], f, indent=2)
        
        # Save mechanistic results
        with open(output_dir / "mechanistic_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.mechanistic_results], f, indent=2)
        
        # Save intervention results
        with open(output_dir / "intervention_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.intervention_results], f, indent=2)
        
        # Save transfer results
        with open(output_dir / "transfer_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.transfer_results], f, indent=2)
        
        # Save defense results
        with open(output_dir / "defense_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.defense_results], f, indent=2)
