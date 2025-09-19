# AIML-project-cia3: Graph Neural Networks on Planetoid

## 1) Project Kick-off
- **Goal**: Node classification on Planetoid datasets (Cora/CiteSeer/PubMed).
- **Success criteria**: Achieve strong test accuracy; report Accuracy and Macro-F1.

## 2) Data Preparation
- **Datasets**: Auto-downloaded `Planetoid` (Cora, CiteSeer, PubMed) on first run.
- **Preprocess & Augment**:
  - Gaussian feature noise (`--noise`)
  - Feature dropout (`--feature-dropout`)
  - Edge dropout (`--edge-dropout`)

## 3) Model Design
- **Variants**: `GCN` and `GAT` implemented in `main.py`.
- **Planning knobs**: hidden size (`--hidden-channels`), dropout (`--dropout`).

## 4) Model Implementation
- **Framework**: PyTorch + PyTorch Geometric.
- **Training**: Clean or augmented features. Evaluate on clean by default.
- **Optional**: Evaluate on noisy features (`--eval-noisy`, `--eval-noise`).

## 5) Evaluation & Tuning
- **Metrics**: Accuracy, Macro-F1 (requires scikit-learn).
- **Grid search**: `--grid-search --gs-hidden ... --gs-lr ... --gs-dropout ...`.

## 6) Deployment Preparation (Demo)
- **Notebook**: `demo.ipynb` shows config, training, eval, and optional tuning.

## 7) Project Report (How to reproduce)
- Document key results, challenges, and improvements below. Example commands:

```bash
# Create venv (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Quick run (GCN on Cora, CPU)
python main.py --dataset Cora --model gcn --epochs 50 --cpu --log-every 10

# With augmentations
python main.py --dataset Cora --model gat --noise 0.1 --feature-dropout 0.1 --edge-dropout 0.05 --epochs 100 --cpu

# Clean vs noisy evaluation
python main.py --dataset Cora --model gcn --epochs 20 --cpu --eval-noisy --eval-noise 0.2

# Grid search
python main.py --grid-search --gs-hidden 16 32 --gs-lr 0.01 0.005 --gs-dropout 0.5 0.6 --cpu
```

## Notes
- First run downloads datasets to `./data/<dataset>`.
- If GPU is available, omit `--cpu` to use CUDA automatically.
- Optional torch-scatter/sparse/cluster/spline wheels may be needed for some ops; PyG will work for this example via pip wheels.
