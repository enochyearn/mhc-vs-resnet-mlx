# Surviving Depth: Stable Hyper-Connections (mHC) vs ResNet

This repo compares three deep networks in MLX:
- ResNet baseline
- Hyper-Connections without constraints (hc_naive)
- Manifold-constrained Hyper-Connections (mHC via Sinkhorn)

The goal is to show that unconstrained dense skip connections can destabilize deep models,
while the Sinkhorn-constrained version stays stable.

Reference paper: https://arxiv.org/pdf/2512.24880

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Run a single mode
python main.py --mode mhc --depth 100

# Run all modes and save comparison plots
python main.py --compare --depth 100
```

## What to Expect

When you run the comparison, you will observe:

1. **ResNet**: Stable learning, loss decreases steadily.
2. **Naive Hyper-Connections**: Gradients likely explode (NaN) or vanish, causing loss to flatline or diverge.
3. **mHC**: Learning is stable (like ResNet) despite dynamic, learnable connections between all layers.

Plots are saved to `results/`:
- `comparison_plot.png`
- `mhc_mixing_matrix.png` (only for mHC)
