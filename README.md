# Surviving Depth: Stable Hyper-Connections (mHC) vs ResNet

This repo compares **three deep networks** on **Fashion-MNIST** (MLX):

- **ResNet** — standard residual baseline
- **hc_naive** — dense Hyper-Connections with **positive weights + per-layer L1 normalization**
- **mHC** — dense Hyper-Connections with **causal Sinkhorn routing** (approx doubly-stochastic)

Reference paper: https://arxiv.org/pdf/2512.24880

---

## Why this exists

Dense skip-style routing *can* help deep models, but it can also become unstable when depth increases.
The core idea here is:

- **hc_naive** only enforces a *local* constraint (each layer mixes past states using a simplex / L1-normalized mixture)
- **mHC** enforces a *global* structure (Sinkhorn makes the full mixing map approximately doubly-stochastic),
  then applies it causally (only using the prefix available at each depth)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

### Compare all models (recommended)

```bash
python main.py --compare --depth 100 --steps 500 --width 64 --batch-size 64 --seed 42
```

### Run a single model

```bash
python main.py --mode mhc --depth 100 --steps 500 --width 64 --batch-size 64 --seed 42
```

### Available flags

* `--mode {resnet,hc_naive,mhc}`
* `--compare` (runs all modes)
* `--depth` (# residual blocks / layers)
* `--steps` (training steps)
* `--width` (channel width)
* `--batch-size`
* `--seed`

---

## Outputs (saved to `results/`)

After running `--compare`, you’ll get:

* `comparison_plot.png`
  Training curves:

  * training loss
  * test accuracy
  * **gradient norm (Layer 0 weight, logged every 10 steps)**

* `mhc_mixing_matrix.png` *(mHC only)*
  Heatmap of the learned Sinkhorn routing matrix

* `metrics_YYYYMMDD-HHMMSS.json`
  Full structured run log containing:

  * `config`: depth / steps / width / batch / seed
  * `summary`: final loss / final acc / max grad norm / diverged
  * `history`: per-step series for loss, test acc, grad norm

* `{mode}_depth{depth}_width{width}_seed{seed}.safetensors`
  Saved weights per model (if training did not diverge)

---

## Example Results

Run:

```bash
python main.py --compare --depth 100 --steps 500 --width 64 --batch-size 64 --seed 42
```

From `results/metrics_20260119-145332.json`:

| Model    | Final Loss | Final Test Acc | Max Grad Norm | Diverged |
| -------- | ---------: | -------------: | ------------: | :------: |
| ResNet   |     0.8477 |         70.19% |        0.7720 |    No    |
| hc_naive |     0.7155 |         72.19% |        0.9872 |    No    |
| mHC      |     0.6923 |         73.24% |        0.7273 |    No    |

**Takeaway:** in this run, **mHC is stable and best-performing**.

> Note: `hc_naive` is still “less structured” than mHC — it may train fine (like here),
> but can also be less stable depending on depth / seed / settings.

---

## What’s the actual constraint difference?

### hc_naive (simplex / L1-normalized mixing)

For each layer `i`, we take positive weights and normalize over the *available prefix*:

* `W = exp(mixing_logits)`
* `weights_i = W[i, :i+1] / sum(W[i, :i+1])`

So each layer forms a **convex mixture** over past states.

### mHC (causal Sinkhorn)

mHC first projects the full matrix using Sinkhorn-Knopp:

* `W = Sinkhorn(exp(mixing_logits))`
  (approximately **row sums ≈ 1** and **col sums ≈ 1**)

Then the forward pass still uses the **causal prefix** for layer `i`:

* `weights_i = W[i, :i+1] / sum(W[i, :i+1])`

This encourages a globally balanced routing structure while staying causal.

When running mHC, you’ll also see a check like:

* `Max Row Sum ≈ 1.0`
* `Max Col Sum ≈ 1.0`

---

## Notes

* `comparison_plot.png` uses the **gradient norm of Layer 0 weights** as a cheap stability signal
  (it is not the norm of *all* gradients in the model).
* Test accuracy is evaluated every 50 steps; intermediate points hold the last evaluated value.
* ResNet is usually fastest; `hc_naive` and `mHC` are slower due to mixing computation.
