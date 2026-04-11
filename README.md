# PSN-2: Perceptual-Symbolic Network v2

A developmental cognitive architecture implementing staged learning from sensorimotor grounding through abstract reasoning.

## 🚀 Quick Start (Kaggle)

```bash
# Sequential training D1 → D6 (recommended)
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D1 \
  --end-stage D6
```

## 📋 What's New

**Latest Update**: Fixed D1 gate failure by implementing batch-type-aware phase selection
- Graph batches now always use `compositional` phase (was incorrectly using `perceptive`)
- Expected improvement: Relation prediction from 1.5% → >80%
- See `docs/guides/FIX_COMPLETE.md` for details

## 🏗️ Architecture Overview

PSN-2 implements a biologically-inspired cognitive architecture with:

- **Vector Symbolic Architectures (VSA)**: High-dimensional distributed representations
- **Attractor Dynamics**: Stable pattern completion and memory retrieval
- **Episodic Replay System (ERS)**: Multi-tier memory with utility-based consolidation
- **Developmental Stages**: Progressive capability emergence (D1-D6)
- **Phase-Based Processing**: Perceptive → Compositional → Recursive

## 📊 Developmental Stages

| Stage | Focus | Steps | Primary Dataset | Key Gates |
|-------|-------|-------|----------------|-----------|
| **D1** | Sensorimotor Grounding | 20K | ARC-AGI-2 (60%) + Graphs (40%) | Object tracking ≥90%, Causal error <20% |
| **D2** | Causal Reasoning | 20K | ARC-AGI-2 (60%) + Graphs (40%) | Causal intervention, Analogies |
| **D3** | Theory of Mind | 15K | ToM/ToMi (60%) + Graphs (40%) | Goal inference, False belief |
| **D4** | Language Grounding | 25K | Wikitext (60%) + ARC (40%) | Linguistic fidelity, Grounding |
| **D5** | Transfer Learning | 30K | GSM8K (60%) + ARC (40%) | Transfer, Meta-learning |
| **D6** | Full Integration | 40K | BBH (60%) + Mixed (40%) | Full system integration |

**Total**: 150,000 steps (~15 hours on 2x GPU)

## 📁 Project Structure

```
psn2_kaggle_full_repo/
├── psn2/                    # Core architecture
│   ├── core.py              # Main PSN2System
│   ├── dc/                  # Developmental Curriculum (D1-D6)
│   ├── ce/                  # Curiosity Engine
│   ├── tae/                 # Temporal Abstraction Engine
│   ├── usl/                 # Universal Symbolic Language
│   └── ess/                 # Emotional Shape System
├── train.py                 # Single-stage training
├── train_sequential.py      # Sequential D1-D6 training
├── evaluate.py              # Evaluation & gate checking
├── configs/                 # Configuration files
├── data/                    # Training datasets
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
    ├── guides/              # Training guides
    └── archive/             # Historical docs
```

## 🎯 Training Commands

### Full Sequential Training
```bash
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D1 \
  --end-stage D6
```

### Resume from Checkpoint
```bash
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D1 \
  --resume /kaggle/working/artifacts/latest.pt
```

### Skip Gate Checks (Not Recommended)
```bash
python train_sequential.py \
  --config configs/default.json \
  --checkpoint-dir /kaggle/working/artifacts \
  --start-stage D1 \
  --skip-gate-check
```

### Single Stage Training
```bash
python train.py \
  --config configs/default.json \
  --stage D1 \
  --steps 20000 \
  --checkpoint-dir /kaggle/working/artifacts
```

## 📈 Monitoring Training

Watch for these indicators:
```
step=X stage=D1 phase=compositional loss=Y pred=Z nodes=256/256 attractors=N goals=M
```

- **Phase**: Should match batch type (graph→compositional, arc→perceptive)
- **Loss**: Should decrease steadily
- **Attractors**: Should grow over time
- **Goals**: Curiosity-driven exploration targets

## ✅ Gate Certification

Each stage must pass gates before progressing:

**D1 Gates:**
- Object tracking accuracy ≥ 90%
- Causal prediction error < 20%
- Temporal trace persistence > 5 pulses
- VSA binding accuracy > 90%

**D2-D6 Gates:** See `docs/guides/` for details

## 🔧 Configuration

Key settings in `configs/default.json`:
```json
{
  "vsa_dim": 512,           # VSA dimensionality
  "max_nodes": 256,         # Maximum network nodes
  "batch_size": 32,         # Training batch size
  "lr": 0.0003,             # Learning rate
  "grid_size": 8,           # ARC grid size
  "grid_vocab": 10,         # Grid vocabulary size
  "rel_vocab_size": 64      # Relational vocabulary size
}
```

## 📚 Documentation

- **Training Guide**: `docs/guides/RETRAIN_GUIDE.md`
- **Phase Fix Details**: `docs/guides/FIX_COMPLETE.md`
- **Kaggle Setup**: `docs/guides/KAGGLE_GUIDE.md`
- **Technical Specs**: `PRD.md`

## 🧪 Testing

```bash
# Test phase selection logic
python scripts/test_phase_selection.py

# Analyze phase usage
python scripts/analyze_phase_usage.py

# Test all datasets
python scripts/test_datasets.py
```

## 📦 Requirements

```
torch>=2.0.0
tqdm
```

Install on Kaggle:
```bash
pip install torch tqdm
```

## 🎓 Key Concepts

### Phase-Based Processing
- **Perceptive**: Spatial pattern recognition (ARC grids)
- **Compositional**: Relational reasoning (graphs, ToM)
- **Recursive**: Sequential reasoning (language, math)

### Vector Symbolic Architectures
- High-dimensional (512D) distributed representations
- Binding/unbinding operations for structured knowledge
- Permutation-based role encoding

### Attractor Dynamics
- Mean-field potential (MPF) basins for stable memories
- Separation-based organization
- Pattern completion and retrieval

## 🐛 Troubleshooting

**D1 gates failing?**
- Check relation prediction (should be >80%, not 1.5%)
- Verify graph batches use `compositional` phase
- See `docs/guides/FIX_COMPLETE.md`

**Training too slow?**
- Reduce `batch_size` in config
- Set `max_arc2_samples` to limit dataset size
- Use fewer stages: `--end-stage D3`

**Out of memory?**
- Reduce `vsa_dim` to 256
- Reduce `max_nodes` to 128
- Reduce `batch_size` to 16

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built for the ARC-AGI Kaggle competition. Implements concepts from cognitive science, neuroscience, and developmental psychology.
