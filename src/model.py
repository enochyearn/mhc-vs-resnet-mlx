import math

import mlx.core as mx
import mlx.nn as nn


def sinkhorn_knopp(log_matrix, iters=5, eps=1e-6):
    """Project log-space weights to a doubly stochastic matrix."""
    log_matrix = mx.minimum(mx.maximum(log_matrix, -10.0), 10.0)
    m = mx.exp(log_matrix)
    for _ in range(iters):
        m = m / (mx.sum(m, axis=-1, keepdims=True) + eps)
        m = m / (mx.sum(m, axis=-2, keepdims=True) + eps)
    return m


def softmax(x, axis=-1, eps=1e-9):
    x = x - mx.max(x, axis=axis, keepdims=True)
    e = mx.exp(x)
    return e / (mx.sum(e, axis=axis, keepdims=True) + eps)


def sigmoid(x):
    return 1.0 / (1.0 + mx.exp(-x))


def rmsnorm(x, eps=1e-6):
    return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


class ResBlock(nn.Module):
    """Pre-activation residual block: GN -> ReLU -> Conv -> GN -> ReLU -> Conv."""

    def __init__(self, channels, dropout_p=0.1):
        super().__init__()
        groups = min(32, max(1, channels // 4))
        self.norm1 = nn.GroupNorm(num_groups=groups, dims=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.norm2 = nn.GroupNorm(num_groups=groups, dims=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x):
        y = nn.relu(self.norm1(x))
        y = self.conv1(y)
        if self.dropout is not None:
            y = self.dropout(y)
        y = nn.relu(self.norm2(y))
        y = self.conv2(y)
        return y


class DeepRunner(nn.Module):
    def __init__(
        self,
        num_layers=50,
        width=64,
        mode="resnet",
        sinkhorn_iters=20,
        streams=1,
        dropout_p=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.width = width  # channel width
        self.mode = mode
        self.sinkhorn_iters = sinkhorn_iters
        self.streams = 1 if mode == "resnet" else max(1, streams)

        self.stem = nn.Conv2d(1, width, kernel_size=3, stride=2, padding=1)
        self.blocks = [ResBlock(width, dropout_p=dropout_p) for _ in range(num_layers)]

        groups = min(32, max(1, width // 4))
        self.norm_final = nn.GroupNorm(num_groups=groups, dims=width)
        self.head = nn.Linear(width, 10)

        self._init_weights()
        if self.mode in ["hc_causal", "mhc_causal", "hc", "mhc"]:
            self._init_mixing_matrix()
        if self.mode in ["hc", "mhc"]:
            self._init_stream_io()
            self._init_dynamic_gating(alpha_init=0.01)

    def _he_init(self, weight):
        if weight.ndim < 2:
            return weight
        if weight.ndim == 2:
            fan_in = weight.shape[1]
        else:
            fan_in = math.prod(weight.shape[1:])
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

    def _init_mixing_matrix(self):
        if self.mode in ["hc_causal", "mhc_causal"]:
            diagonal_bias = mx.eye(self.num_layers) * 4.0
            noise = mx.random.normal((self.num_layers, self.num_layers)) * 0.05
            self.mixing_logits = diagonal_bias + noise
            return

        if self.mode in ["hc", "mhc"]:
            eye = mx.eye(self.streams)
            base = eye * 5.0 + (1.0 - eye) * (-5.0)
            noise = mx.random.normal(
                (self.num_layers, self.streams, self.streams)
            ) * 0.01
            self.mixing_logits = base[None, :, :] + noise

    def _init_stream_io(self):
        self.pre_logits = mx.full((self.num_layers, self.streams), -5.0)
        self.post_logits = mx.full((self.num_layers, self.streams), -5.0)
        self.readout_logits = mx.full((self.streams,), -5.0)

        self.pre_logits[:, 0] = 5.0
        self.post_logits[:, 0] = 5.0
        self.readout_logits[0] = 5.0

    def _init_dynamic_gating(self, alpha_init=0.01):
        d = self.streams * self.width
        scale = 1.0 / math.sqrt(d)
        self.phi_pre = mx.random.normal((self.num_layers, d, self.streams)) * scale
        self.phi_post = mx.random.normal((self.num_layers, d, self.streams)) * scale
        self.phi_res = (
            mx.random.normal((self.num_layers, d, self.streams * self.streams)) * scale
        )

        self.alpha_pre = mx.ones((self.num_layers, 1)) * alpha_init
        self.alpha_post = mx.ones((self.num_layers, 1)) * alpha_init
        self.alpha_res = mx.ones((self.num_layers, 1)) * alpha_init

    def _mixing_matrix(self):
        if self.mode == "mhc_causal":
            return sinkhorn_knopp(self.mixing_logits, iters=self.sinkhorn_iters)
        if self.mode == "hc_causal":
            return mx.exp(self.mixing_logits)
        return None

    def _stream_weights(self, layer_idx):
        return None

    def _apply_block(self, block, x):
        if x.ndim == 4:
            return block(x)
        if x.ndim == 5:
            batch, streams, height, width, channels = x.shape
            flat = mx.reshape(x, (batch * streams, height, width, channels))
            out = block(flat)
            return mx.reshape(out, (batch, streams, height, width, channels))
        raise ValueError(f"Unexpected input rank: {x.ndim}")

    def _mix_streams(self, x, weights):
        if weights.ndim == 2:
            mixed = mx.tensordot(weights, x, axes=([1], [1]))
            return mx.transpose(mixed, (1, 0, 2, 3, 4))
        if weights.ndim == 3:
            return mx.sum(
                weights[:, :, :, None, None, None] * x[:, None, :, :, :, :], axis=2
            )
        raise ValueError(f"Unexpected weights shape: {weights.shape}")

    def _compute_stream_mappings(self, layer_idx, current_state):
        pooled = mx.mean(current_state, axis=(2, 3))
        x_vec = mx.reshape(pooled, (pooled.shape[0], self.streams * self.width))
        x_norm = rmsnorm(x_vec)

        pre_dyn = mx.matmul(x_norm, self.phi_pre[layer_idx])
        post_dyn = mx.matmul(x_norm, self.phi_post[layer_idx])
        res_dyn = mx.matmul(x_norm, self.phi_res[layer_idx])
        res_dyn = mx.reshape(res_dyn, (x_norm.shape[0], self.streams, self.streams))

        tilde_pre = self.alpha_pre[layer_idx] * pre_dyn + self.pre_logits[layer_idx]
        tilde_post = self.alpha_post[layer_idx] * post_dyn + self.post_logits[layer_idx]
        tilde_res = self.alpha_res[layer_idx] * res_dyn + self.mixing_logits[layer_idx]

        if self.mode == "mhc":
            h_pre = sigmoid(tilde_pre)
            h_post = 2.0 * sigmoid(tilde_post)
            h_res = sinkhorn_knopp(tilde_res, iters=self.sinkhorn_iters)
        else:
            h_pre, h_post, h_res = tilde_pre, tilde_post, tilde_res

        return h_pre, h_post, h_res

    def _expand_to_streams(self, x):
        if self.streams == 1:
            return x[:, None, ...]
        batch, height, width, channels = x.shape
        zeros = mx.zeros((batch, self.streams - 1, height, width, channels))
        return mx.concatenate([x[:, None, ...], zeros], axis=1)

    def __call__(self, x):
        x = self.stem(x)

        current_state = x

        if self.mode == "resnet":
            for i in range(self.num_layers):
                residual = self.blocks[i](current_state)
                current_state = current_state + residual
        elif self.mode in ["hc_causal", "mhc_causal"]:
            x = mx.repeat(x[:, None, ...], self.streams, axis=1)
            history = [x]
            mix_mat = self._mixing_matrix()
            for i in range(self.num_layers):
                weights = mix_mat[i, : len(history)]
                weights = weights / (mx.sum(weights) + 1e-6)
                inp = history[0] * weights[0]
                for j in range(1, len(history)):
                    inp = inp + (history[j] * weights[j])

                residual = self._apply_block(self.blocks[i], inp)
                current_state = inp + residual
                history.append(current_state)
        elif self.mode in ["hc", "mhc"]:
            current_state = self._expand_to_streams(x)
            for i in range(self.num_layers):
                h_pre, h_post, h_res = self._compute_stream_mappings(i, current_state)

                x_in = mx.sum(
                    current_state * h_pre[:, :, None, None, None], axis=1
                )
                delta = self.blocks[i](x_in)

                mixed = self._mix_streams(current_state, h_res)
                write_in = delta[:, None, ...] * h_post[:, :, None, None, None]
                current_state = mixed + write_in

        if current_state.ndim == 5:
            if self.mode in ["hc", "mhc"]:
                readout_w = sigmoid(self.readout_logits)
                current_state = mx.sum(
                    current_state * readout_w[None, :, None, None, None], axis=1
                )
            else:
                current_state = mx.mean(current_state, axis=1)

        x = nn.relu(self.norm_final(current_state))
        x = mx.mean(x, axis=(1, 2))
        return self.head(x)
