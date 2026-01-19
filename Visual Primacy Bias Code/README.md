# Visual Primacy Bias: Experiment Code

Official implementation for the paper:

**"Visual Primacy Bias: Overriding Safety Alignment via Cross-Modal Attention Hijacking in Multimodal LLMs"**

## Abstract

We identify Visual Primacy Bias (VPB): when MLLMs receive conflicting instructions between textual safety prompts and visual inputs containing malicious commands, they consistently prioritize visual signals. We propose Typographic Attention Steering (TAS), a training-free attack framework that exploits VPB through gradient-guided typographic optimization.

## Repository Structure

```
├── configs/              # Configuration files
│   └── default.yaml      # Default experiment configuration
├── src/
│   ├── models/           # MLLM interfaces
│   │   ├── interface.py  # Abstract base class
│   │   ├── llava.py      # LLaVA-Next implementation
│   │   ├── qwen.py       # Qwen-VL implementation
│   │   ├── instructblip.py # InstructBLIP implementation
│   │   └── gpt4v.py      # GPT-4V API wrapper
│   ├── rendering/        # TAS image rendering
│   │   ├── pipeline.py   # Main rendering pipeline
│   │   └── background.py # Background generation
│   ├── scoring/          # Gradient-guided selection
│   │   └── gradient_scorer.py
│   ├── analysis/         # Mechanistic analysis
│   │   ├── attention.py  # Attention ratio analysis
│   │   ├── attribution.py # IG, QKV ablation, Logit Lens
│   │   └── transfer.py   # Transfer attack analysis
│   ├── evaluation/       # Evaluation metrics
│   │   ├── asr.py        # Attack Success Rate (GPT-4 judge)
│   │   └── ocr.py        # OCR evasion evaluation
│   ├── baselines/        # Baseline attacks
│   │   ├── gcg.py        # GCG text attack
│   │   ├── pgd.py        # PGD visual noise
│   │   ├── figstep.py    # FigStep attack
│   │   ├── umk.py        # UMK attack
│   │   └── direct.py     # Direct instruction baseline
│   ├── defenses/         # Defense implementations
│   │   ├── input_level.py   # OCR filter, blur, compression
│   │   ├── prompt_level.py  # Safety prompts, keyword filter
│   │   └── model_level.py   # Output filter, attention rebalancing
│   ├── data/             # Dataset loaders
│   │   ├── mm_safetybench.py
│   │   └── jailbreakv.py
│   ├── experiments/      # Experiment runner
│   │   └── runner.py
│   └── visualization/    # Plotting utilities
│       ├── attention.py  # Attention visualizations
│       ├── tsne.py       # t-SNE modality gap
│       ├── defense.py    # Defense comparison plots
│       └── latex_export.py # LaTeX table generation
├── scripts/              # Experiment scripts
│   ├── run_attack.py     # Main TAS attack
│   ├── run_analysis.py   # Mechanistic analysis
│   ├── run_defense_eval.py # Defense evaluation
│   └── generate_results.py # Generate paper tables/figures
├── tests/                # Unit tests
├── data/                 # Dataset directory (download separately)
├── checkpoints/          # Experiment checkpoints
└── outputs/              # Results output directory
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Install optional dependencies for full functionality
pip install pytesseract paddleocr sentence-transformers
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (recommended: A100 80GB for white-box experiments)
- API keys for GPT-4V evaluation (set `OPENAI_API_KEY`)

## Quick Start

### 1. Run TAS Attack Experiment

```bash
python scripts/run_attack.py \
    --config configs/default.yaml \
    --model llava-next-7b \
    --num-samples 100
```

### 2. Run Mechanistic Analysis

```bash
python scripts/run_analysis.py \
    --config configs/default.yaml \
    --analysis-type attention
```

### 3. Evaluate Defenses

```bash
python scripts/run_defense_eval.py \
    --config configs/default.yaml \
    --defense ocr_filter
```

### 4. Generate Paper Results

```bash
python scripts/generate_results.py \
    --results-dir outputs/results \
    --output-dir outputs/paper
```

## Configuration

See `configs/default.yaml` for all configurable parameters:

- **Models**: LLaVA-Next, Qwen-VL, InstructBLIP, GPT-4V
- **Attack**: num_candidates (120), lambda_weight (0.3)
- **Rendering**: 15 fonts, distortion ranges
- **Evaluation**: ASR threshold, OCR engines

## API Keys

```bash
# For GPT-4V evaluation
export OPENAI_API_KEY="your-key"

# For Google Cloud Vision OCR (optional)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

## Running Tests

```bash
pytest tests/ -v
```

## Datasets

Download datasets separately:
- MM-SafetyBench: https://github.com/isXinLiu/MM-SafetyBench
- JailBreakV-28K: https://huggingface.co/datasets/JailbreakV-28K

Place in `data/` directory following the loader conventions.

## Citation

```bibtex
@inproceedings{visualprimacybias2026,
  title={Visual Primacy Bias: Overriding Safety Alignment via Cross-Modal Attention Hijacking in Multimodal LLMs},
  author={Anonymous},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This code is released under the MIT License for academic research purposes.

