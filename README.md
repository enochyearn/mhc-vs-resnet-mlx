# Surviving Depth: Stable Hyper-Connections (mHC) vs ResNet

This repo compares **five deep networks** on **Fashion-MNIST** (MLX):

- **resnet** — standard residual baseline
- **hc_causal** — dense Hyper-Connections with **positive weights + per-layer L1 normalization** (causal prefix)
- **mhc_causal** — dense Hyper-Connections with **causal Sinkhorn routing** (approx doubly-stochastic)
- **hc** — paper-style stream readout/write-in HC (meaningful when `--streams > 1`)
- **mhc** — paper-style stream readout/write-in mHC (meaningful when `--streams > 1`)

Reference paper: https://arxiv.org/pdf/2512.24880

ResNet ignores `--streams`; stream modes are meaningful when `--streams > 1`.
When `--streams = 1`, doubly-stochastic constraints degenerate to the identity case,
so mHC behaves like a standard residual mapping.

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

## What Causal Means Here

Causal here means **depth-prefix mixing**: at layer `i`, you can only mix from states
`0..i` (never from "future" layers). This is **not** autoregressive time causality.

Intuition:

- **Causal HC/mHC ≈ DenseNet-like depth history mixing**
- **Stream hc/mhc ≈ multi-stream residual routing with readout/write-in (Eq. 3)**

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

ResNet ignores `--streams`; stream modes are meaningful when `--streams > 1`.

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

## Benchmark Results (Jan 25, 2026)

### Depth 100

```bash
python main.py --compare --depth 100 --steps 500 --width 64 --batch-size 64 --seed 42
```

From `results/metrics_20260125-122941.json`:

| Model      | Final Loss | Final Test Acc | Diverged | Wall Time |
| ---------- | ---------: | -------------: | :------: | --------: |
| resnet     |     0.8842 |         64.93% |    No    |     3m40s |
| hc_causal  |     0.8663 |         66.16% |    No    |     7m24s |
| mhc_causal |     0.8370 |         67.25% |    No    |     7m26s |
| hc         |        NaN |          9.95% |  **Yes** |    <1 min |
| mhc        |     0.8601 |         65.64% |    No    |     4m09s |

### Depth 500

```bash
python main.py --compare --depth 500 --steps 2000 --width 32 --batch-size 32 --seed 42
```

From `results/metrics_20260125-191509.json`:

| Model      | Final Loss | Final Test Acc | Diverged | Wall Time |
| ---------- | ---------: | -------------: | :------: | --------: |
| resnet     |     1.0179 |         63.63% |    No    |    32m05s |
| hc_causal  |     0.5125 |         79.67% |    No    |  2h44m56s |
| mhc_causal |     0.5237 |         79.27% |    No    |  2h42m32s |
| hc         |        NaN |          9.98% |  **Yes** |    ~1 min |
| mhc        |     0.9359 |         71.43% |    No    |    44m44s |

mHC also satisfies the Sinkhorn routing constraint (row/col sums ≈ 1.0).

## Why does hc diverge but mhc doesn't?

**HC is unconstrained** and can amplify signals/gradients catastrophically. **mHC**
projects `Hres` onto the doubly-stochastic manifold (Sinkhorn) and constrains
`Hpre/Hpost` via sigmoid, which keeps norms bounded and stabilizes deep propagation.

## Model Stats (Compute / Memory)

### Complexity cheat sheet (high-level)

| Mode            | Extra Params      | Extra Compute                  | Extra Memory           | Notes                              |
| --------------- | ----------------- | ------------------------------ | ---------------------- | ---------------------------------- |
| resnet          | baseline          | **O(L)**                       | **O(L)** activations   | fastest                            |
| hc_causal       | + mixing logits   | **O(L²)** depth mixing         | **O(L)** store history | prefix mixing                      |
| mhc_causal      | + mixing logits   | **O(L² · T)** (Sinkhorn iters) | **O(L)**               | global DS projection + causal use  |
| hc (streams=n)  | + Hpre/Hpost/Hres | **O(L · n²)** mixing + compute | **O(n)** streams       | unconstrained → often diverges     |
| mhc (streams=n) | + Hpre/Hpost/Hres | **O(L · n² · T)**              | **O(n)**               | Sinkhorn stabilizes stream routing |

## Training Environment / Reproducibility

- Runs shown use MLX on Apple Silicon; exact timing depends on hardware, batch size,
  compile mode, and stream count.
- `mx.compile` is on by default; use `--no-compile` for easier timing comparisons.

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

When running mhc_causal or mhc, you'll also see a check like:

* `Max Row Sum ≈ 1.0`
* `Max Col Sum ≈ 1.0`

### hc (stream readout/write-in, unconstrained)

For each layer, we read out a single stream, run the block once, and write back:

* `x_in = sum_s Hpre_l[s] * x_s`
* `delta = F(x_in)`
* `x = Hres_l x + Hpost_l^T * delta`

`Hpre`, `Hpost`, and `Hres` are **input-dependent** functions of the current state
via `RMSNorm(vec(x_l))` and linear projections.

### mhc (stream readout/write-in + Sinkhorn)

Same readout/write-in as above, but with Sinkhorn-constrained `Hres_l`.
For mHC, the dynamic mappings are constrained as:

* `Hpre = sigmoid(H̃pre)`
* `Hpost = 2 * sigmoid(H̃post)`
* `Hres = Sinkhorn(H̃res)`

---

## Notes

* `comparison_plot.png` uses the **gradient norm of Layer 0 weights** as a cheap stability signal
  (it is not the norm of *all* gradients in the model).
* Test accuracy is evaluated every 1000 steps; intermediate points hold the last evaluated value.
* ResNet is usually fastest; causal and stream HC variants are slower due to mixing computation.
---

## Roadmap / TODO

### Training recipe improvements (make baselines stronger)

- [x] Tune **LR schedule** (warmup + cosine or step decay)
- [x] Tune **weight decay** (AdamW-style)
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
