# DualSTGF: Dual-View Spectral-Temporal Graph Fusion

<p align="center">
  <img src="Architecture.png" alt="DualSTAGE Architecture" width="800"/>
</p>

A specialized framework for **incipient fault detection** in complex industrial systems using **Dynamic Graph Neural Networks**.

This repository implements **DualSTGF** (Dual Spectral-Temporal Graph Fusion). It learns two concurrent graph topologies:
1.  **Temporal Graph ($A_{time}$)**: Captures dynamic correlations (nodes moving together).
2.  **Spectral Graph ($A_{freq}$)**: Captures frequency-domain similarities (nodes resonating together).

By monitoring the **structural divergence** between these two graphs ($D_{div}$), the model detects incipient faults (wear, drift, fouling) *before* they manifest as gross reconstruction errors.

---

## 1. Key Features

*   **Dual-View Learning**: Simultaneous `GRUEncoder` (time) and `SpectralEncoder` (frequency) branches.
*   **Structural Divergence Score**: Explicitly measures mismatch between physical connectivity and spectral behavior (early warning signal).
*   **Topology-Aware Anomaly Scoring**: Penalizes errors on central nodes more heavily.
*   **Interactive Visualization**: Plotly-based dashboards for reconstruction and anomaly score analysis.

---

## 2. Supported Datasets

Data adapters are defined in `datasets/` and accessed via `--dataset-key`.

| Dataset Key | Name | Type | Source |
| :--- | :--- | :--- | :--- |
| `pronto` | PRONTO Benchmark | Multiphase Flow | [Zenodo](https://zenodo.org/records/1341583) |
| `ashrae` | ASHRAE 1043-RP | HVAC/Refrigeration | Research Project |

### 2.1 Data Setup
Place datasets in `data/`:
```bash
DualSTAGE/
├── data/
│   ├── pronto/            # PRONTO benchmark files
│   └── ASHRAE_1043_RP/    # ASHRAE CSV files
```

---

## 3. Training

### 3.1 Minimal Training
```bash
python train.py \
    --dataset-key pronto \
    --data-dir data/pronto \
    --epochs 50 \
    --batch-size 32 \
    --window-size 60 \
    --lr 1e-3
```

### 3.2 Full Training (Dual-View Mode)
Use `evaluation/train_dualstage.py` for the full training pipeline with spectral view, TEA evaluation, and checkpointing.

```bash
python evaluation/train_dualstage.py \
    --dataset-key pronto \
    --use-spectral-view \
    --freq-embed-dim 16 \
    --freq-band-mix mlp \
    --lambda-div 0.1 \
    --anomaly-weight 0.5 \
    --epochs 20 \
    --batch-size 64 \
    --use-amp
```

### Key Arguments
*   `--use-spectral-view`: Enables the spectral branch.
*   `--freq-embed-dim`: Dimension of frequency embeddings (default: 16).
*   `--freq-band-mix`: Method to mix frequency bins (`none`, `conv`, `mlp`).
*   `--lambda-div`: Weight for the **Structural Divergence Loss** (crucial for incipient detection).
*   `--anomaly-weight`: Weight for the Topology-Aware reconstruction penalty.

---

## 4. Repository Structure

*   `dualstage/src/model/dualstage.py`: Core model (Dual-View Architecture).
*   `train.py`: Minimal training script.
*   `evaluation/train_dualstage.py`: Full training loop with divergence loss, TEA, and checkpointing.
*   `datasets/`: Data adapters for PRONTO and ASHRAE benchmarks.

## 5. Quick Start
1.  Install dependencies: `pip install -r requirements.txt`
2.  Download PRONTO or ASHRAE data.
3.  Run training: `python train.py --dataset-key pronto --data-dir data/pronto --epochs 50 --window-size 60 --lr 1e-3`
