# Surviving Depth: Stable Hyper-Connections (mHC) vs ResNet

This repo compares **five deep networks** on **Fashion-MNIST** (MLX):

- **resnet** — standard residual baseline
- **hc_causal** — dense Hyper-Connections with **positive weights + per-layer L1 normalization** (causal prefix)
- **mhc_causal** — dense Hyper-Connections with **causal Sinkhorn routing** (approx doubly-stochastic)
- **hc** — paper-style stream readout/write-in HC (requires `--streams > 1`)
- **mhc** — paper-style stream readout/write-in mHC (requires `--streams > 1`)

Reference paper: https://arxiv.org/pdf/2512.24880

ResNet ignores `--streams`; stream modes only become meaningful when `--streams > 1`.

---

## Why this exists

Dense skip-style routing *can* help deep models, but it can also become unstable when depth increases.
The core idea here is:

- **hc_causal** only enforces a *local* constraint (each layer mixes past states using a simplex / L1-normalized mixture)
- **mhc_causal** enforces a *global* structure (Sinkhorn makes the full mixing map approximately doubly-stochastic),
  then applies it causally (only using the prefix available at each depth)

There are two flavors of HC in this repo:

- **Causal** (`hc_causal`, `mhc_causal`) mixes the depth-history prefix at each layer.
- **Stream readout/write-in** (`hc`, `mhc`) does Hpre/Hpost per layer, matching Eq.(3).

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
python main.py --mode mhc_causal --depth 100 --steps 500 --width 64 --batch-size 64 --seed 42
```

### Run a stream readout/write-in model

```bash
python main.py --mode mhc --streams 4 --depth 100 --steps 500 --width 64 --batch-size 64 --seed 42
```

### Available flags

* `--mode {resnet,hc,hc_causal,mhc,mhc_causal}`
* `--compare` (runs all modes)
* `--depth` (# residual blocks / layers)
* `--steps` (training steps)
* `--width` (channel width)
* `--batch-size`
* `--seed`
* `--streams` (default 1)
* `--lr`
* `--weight-decay`
* `--warmup-steps`
* `--no-compile` (disable `mx.compile`)
* `--dropout`
* `--no-schedule` (disable LR schedule)

ResNet ignores `--streams`; stream modes are intended for `--streams > 1`.

---

## Outputs (saved to `results/`)

After running `--compare`, you’ll get:

* `comparison_plot.png`
  Training curves:

  * training loss
  * test accuracy
  * **gradient norm (Layer 0 weight, logged every 50 steps)**

* `mhc_mixing_matrix.png` *(mHC modes only)*
  Heatmap of the learned Sinkhorn routing matrix

* `metrics_YYYYMMDD-HHMMSS.json`
  Full structured run log containing:

  * `config`: depth / steps / width / batch / seed / streams / lr / weight_decay / warmup_steps / compile_step / dropout / use_schedule
  * `summary`: final loss / final acc / max grad norm / diverged
  * `history`: per-step series for loss, test acc, grad norm

* `{mode}_depth{depth}_width{width}_seed{seed}.safetensors`
  Saved weights for ResNet runs (if training did not diverge)
* `{mode}_depth{depth}_width{width}_streams{streams}_seed{seed}.safetensors`
  Saved weights for HC/mHC modes (if training did not diverge)

---

## Example Results

Run:

```bash
python main.py --compare --depth 100 --steps 500 --width 64 --batch-size 64 --seed 42
```

From `results/metrics_20260119-145332.json` (causal modes, `--streams 1`):

| Model      | Final Loss | Final Test Acc | Max Grad Norm | Diverged |
| ---------- | ---------: | -------------: | ------------: | :------: |
| resnet     |     0.8477 |         70.19% |        0.7720 |    No    |
| hc_causal  |     0.7155 |         72.19% |        0.9872 |    No    |
| mhc_causal |     0.6923 |         73.24% |        0.7273 |    No    |

**Takeaway:** in this run, **mhc_causal is stable and best-performing**.

> Note: `hc_causal` is still “less structured” than `mhc_causal` — it may train fine (like here),
> but can also be less stable depending on depth / seed / settings.

---

## Scaling Check (Depth 500)

Run:

```bash
python main.py --compare --depth 500 --steps 2000 --width 32 --batch-size 32 --seed 42
```

From `results/metrics_20260121-034822.json` (causal modes, `--streams 1`):

| Model      | Final Loss | Final Test Acc | Diverged |
| ---------- | ---------: | -------------: | :------: |
| resnet     |     0.7098 |         76.29% |    No    |
| hc_causal  |     0.4415 |         81.01% |    No    |
| mhc_causal |     0.5064 |         81.37% |    No    |

mhc_causal also satisfies the Sinkhorn routing constraint:

- `Max Row Sum ≈ 1.0000`
- `Max Col Sum ≈ 1.0000`

**Takeaway:** at **500 layers**, both Hyper-Connection variants remain stable and reach **~81%** accuracy,
while ResNet lags behind in this setup.

## What’s the actual constraint difference?

### hc_causal (simplex / L1-normalized mixing)

For each layer `i`, we take positive weights and normalize over the *available prefix*:

* `W = exp(mixing_logits)`
* `weights_i = W[i, :i+1] / sum(W[i, :i+1])`

So each layer forms a **convex mixture** over past states.

### mhc_causal (causal Sinkhorn)

mhc_causal first projects the full matrix using Sinkhorn-Knopp:

* `W = Sinkhorn(exp(mixing_logits))`
  (approximately **row sums ≈ 1** and **col sums ≈ 1**)

Then the forward pass still uses the **causal prefix** for layer `i`:

* `weights_i = W[i, :i+1] / sum(W[i, :i+1])`

This encourages a globally balanced routing structure while staying causal.

When running mhc_causal (or mhc), you’ll also see a check like:

* `Max Row Sum ≈ 1.0`
* `Max Col Sum ≈ 1.0`

### hc (stream readout/write-in)

For each layer, we read out a single stream, run the block once, and write back:

* `x_in = sum_s Hpre_l[s] * x_s`
* `delta = F(x_in)`
* `x = Hres_l x + Hpost_l^T * delta`

### mhc (stream readout/write-in + Sinkhorn)

Same readout/write-in as above, but with Sinkhorn-constrained `Hres_l`.

---

## Notes

* `comparison_plot.png` uses the **gradient norm of Layer 0 weights** as a cheap stability signal
  (it is not the norm of *all* gradients in the model).
* Test accuracy is evaluated every 1000 steps; intermediate points hold the last evaluated value.
* ResNet is usually fastest; causal and stream HC variants are slower due to mixing computation.
---

## Roadmap / TODO

### Training recipe improvements (make baselines stronger)

- [ ] Add **LR schedule**: warmup + cosine decay (or step decay)
- [ ] Add **weight decay** (AdamW-style)
- [ ] Run longer: report results at **~10–30 epochs** (current runs are mainly “stability stress tests”)
- [ ] Add basic preprocessing: **mean/std normalization**
- [ ] Optional: light augmentation (random crop / translate)

### Next experiments (research direction)

**P0 — Long-sequence modeling (highest priority)**  
Goal: show mHC as a “stable infinite-memory router” when sequence length is the real enemy.

- [ ] Build a 1D sequence version (depth = time) + switch to a long-sequence dataset:
  - [ ] LRA Pathfinder / Pathfinder-X (1K → 16K) 
  - [ ] LRA ListOps (2K tokens)
  - [ ] Synthetic copy / associative recall for ultra-long horizons (10K+)
- [ ] Compare **RNN/Transformer vs hc_causal vs mhc_causal** as sequence length grows (1k → 10k steps)

**P1 — Deep GNNs (oversmoothing killer)**  
Goal: go from 5-layer GNN limit → 50–200 layers.

- [ ] Replace message passing aggregation with an mHC-style mixing step
- [ ] Try on OGB-Arxiv / OGB-Proteins
- [ ] Metric: accuracy vs depth + oversmoothing diagnostics

**P2 — Flow matching / Neural ODE stability**  
Goal: show “conservative-ish routing” reduces long-horizon drift/explosion.

- [ ] Toy 2D flows first, then scale up
- [ ] Compare ResNet block vs mHC block inside long integration
