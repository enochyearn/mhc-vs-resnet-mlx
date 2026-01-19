from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from .dataset import FashionMNISTLoader
from .model import DeepRunner, sinkhorn_knopp


def results_dir_path():
    return Path(__file__).resolve().parents[1] / "results"


def iter_leaves(tree):
    if isinstance(tree, dict):
        for v in tree.values():
            yield from iter_leaves(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from iter_leaves(v)
    else:
        yield tree


def first_block_grad(grads):
    if isinstance(grads, dict) and "blocks" in grads:
        try:
            return grads["blocks"][0]["weight"]
        except (KeyError, IndexError, TypeError):
            pass
    for leaf in iter_leaves(grads):
        if leaf is not None:
            return leaf
    return None


def run_experiment(
    mode,
    depth,
    steps=200,
    width=64,
    batch_size=64,
    seed=0,
):
    loader = FashionMNISTLoader()
    model = DeepRunner(num_layers=depth, width=width, mode=mode)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=0.001)

    def loss_fn(model, bx, by):
        logits = model(bx)
        return mx.mean(nn.losses.cross_entropy(logits, by))

    step_fn = nn.value_and_grad(model, loss_fn)
    batch_iter = loader.get_batches(
        batch_size, split="train", shuffle=True, seed=seed, repeat=True
    )

    history = {"loss": [], "acc": [], "grad_norm": []}

    for _ in tqdm(range(steps), desc=mode, leave=True):
        bx, by = next(batch_iter)

        loss, grads = step_fn(model, bx, by)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        logits = model(bx)
        acc = mx.mean((mx.argmax(logits, axis=1) == by).astype(mx.float32))
        mx.eval(acc)

        loss_val = float(loss.item())
        acc_val = float(acc.item())

        g0 = first_block_grad(grads)
        if g0 is not None:
            mx.eval(g0)
            grad_norm = float(np.linalg.norm(np.array(g0)))
        else:
            grad_norm = float("nan")

        history["loss"].append(loss_val)
        history["acc"].append(acc_val)
        history["grad_norm"].append(grad_norm)

        if np.isnan(loss_val) or grad_norm > 1e6:
            tqdm.write("DIVERGENCE DETECTED")
            break

    return history, model


def plot_results(histories, results_dir=None):
    results_dir = Path(results_dir) if results_dir else results_dir_path()
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for mode, history in histories.items():
        axes[0].plot(history["loss"], label=mode)
        axes[1].plot(history["acc"], label=mode)
        axes[2].plot(history["grad_norm"], label=mode)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Training Accuracy")
    axes[2].set_title("Gradient Norm (Layer 0)")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    output_path = results_dir / "comparison_plot.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_mhc_matrix(model, results_dir=None):
    if not hasattr(model, "mixing_logits"):
        return None
    results_dir = Path(results_dir) if results_dir else results_dir_path()
    results_dir.mkdir(parents=True, exist_ok=True)

    w_final = sinkhorn_knopp(model.mixing_logits)
    mx.eval(w_final)
    row_sums = mx.sum(w_final, axis=1)
    col_sums = mx.sum(w_final, axis=0)
    mx.eval(row_sums, col_sums)
    row_sums = np.array(row_sums)
    col_sums = np.array(col_sums)

    print("\n[mHC Check] Matrix Constraints:")
    print(f"  Max Row Sum: {row_sums.max():.4f} (Should be ~1.0)")
    print(f"  Max Col Sum: {col_sums.max():.4f} (Should be ~1.0)")

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.array(w_final), cmap="viridis")
    ax.set_title("Learned mHC Routing Map")
    fig.colorbar(im, ax=ax)
    output_path = results_dir / "mhc_mixing_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_comparison(
    depth,
    modes=None,
    steps=200,
    width=64,
    batch_size=64,
    seed=0,
):
    modes = modes or ["resnet", "hc_naive", "mhc"]
    histories = {}
    models = {}

    for mode in modes:
        history, model = run_experiment(
            mode,
            depth,
            steps=steps,
            width=width,
            batch_size=batch_size,
            seed=seed,
        )
        histories[mode] = history
        models[mode] = model

    plot_path = plot_results(histories)
    mhc_path = None
    if "mhc" in models:
        mhc_path = plot_mhc_matrix(models["mhc"])

    return histories, models, plot_path, mhc_path
