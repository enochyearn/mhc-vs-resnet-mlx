import math

import mlx.core as mx
import mlx.nn as nn


def sinkhorn_knopp(log_matrix, iters=5, eps=1e-6):
    """Project log-space weights to a doubly stochastic matrix."""
    m = mx.exp(log_matrix)
    for _ in range(iters):
        m = m / (mx.sum(m, axis=1, keepdims=True) + eps)
        m = m / (mx.sum(m, axis=0, keepdims=True) + eps)
    return m


class ResBlock(nn.Module):
    """Pre-activation residual block: GN -> ReLU -> Conv -> GN -> ReLU -> Conv."""

    def __init__(self, channels):
        super().__init__()
        groups = min(32, max(1, channels // 4))
        self.norm1 = nn.GroupNorm(num_groups=groups, dims=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, dims=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x):
        y = nn.relu(self.norm1(x))
        y = self.conv1(y)
        y = nn.relu(self.norm2(y))
        y = self.conv2(y)
        return y


class DeepRunner(nn.Module):
    def __init__(self, num_layers=50, width=64, mode="resnet", sinkhorn_iters=5):
        super().__init__()
        self.num_layers = num_layers
        self.width = width  # channel width
        self.mode = mode
        self.sinkhorn_iters = sinkhorn_iters

        self.stem = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.blocks = [ResBlock(width) for _ in range(num_layers)]

        groups = min(32, max(1, width // 4))
        self.norm_final = nn.GroupNorm(num_groups=groups, dims=width)
        self.head = nn.Linear(width, 10)

        if self.mode in ["hc_naive", "mhc"]:
            self.mixing_logits = mx.random.normal((num_layers, num_layers)) * 0.01
            if self.mode == "hc_naive":
                self.mixing_logits = self.mixing_logits - 4.0

        self._init_weights()

    def _he_init(self, weight):
        if weight.ndim < 2:
            return weight
        if weight.ndim == 2:
            fan_in = weight.shape[1]
        else:
            fan_in = 1
            for dim in weight.shape[1:]:
                fan_in *= dim
        scale = math.sqrt(2.0 / fan_in)
        return mx.random.normal(weight.shape) * scale

    def _init_weights(self):
        res_scale = 1.0 / math.sqrt(max(1, self.num_layers))

        self.stem.weight = self._he_init(self.stem.weight)
        if self.stem.bias is not None:
            self.stem.bias = mx.zeros_like(self.stem.bias)

        for block in self.blocks:
            block.conv1.weight = self._he_init(block.conv1.weight)
            if block.conv1.bias is not None:
                block.conv1.bias = mx.zeros_like(block.conv1.bias)
            block.conv2.weight = self._he_init(block.conv2.weight) * res_scale
            if block.conv2.bias is not None:
                block.conv2.bias = mx.zeros_like(block.conv2.bias)

        self.head.weight = self._he_init(self.head.weight)
        if self.head.bias is not None:
            self.head.bias = mx.zeros_like(self.head.bias)

    def _mixing_matrix(self):
        if self.mode == "mhc":
            return sinkhorn_knopp(self.mixing_logits, iters=self.sinkhorn_iters)
        if self.mode == "hc_naive":
            return mx.exp(self.mixing_logits)
        return None

    def __call__(self, x):
        x = self.stem(x)

        history = [x]
        mix_mat = self._mixing_matrix()
        current_state = x

        for i in range(self.num_layers):
            if self.mode == "resnet":
                inp = current_state
            else:
                weights = mix_mat[i, : len(history)]
                # Note: This re-stacks history each layer (O(depth^2)); fine for demo scale.
                h_stack = mx.stack(history)
                inp = mx.tensordot(weights, h_stack, axes=([0], [0]))

            residual = self.blocks[i](inp)

            if self.mode == "resnet":
                current_state = current_state + residual
            else:
                current_state = residual

            history.append(current_state)

        x = nn.relu(self.norm_final(current_state))
        x = mx.mean(x, axis=(1, 2))
        return self.head(x)
