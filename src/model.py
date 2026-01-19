import mlx.core as mx
import mlx.nn as nn


def sinkhorn_knopp(log_matrix, iters=5, eps=1e-6):
    """Project log-space weights to a doubly stochastic matrix."""
    m = mx.exp(log_matrix)
    for _ in range(iters):
        m = m / (mx.sum(m, axis=1, keepdims=True) + eps)
        m = m / (mx.sum(m, axis=0, keepdims=True) + eps)
    return m


class DeepRunner(nn.Module):
    def __init__(self, num_layers=100, width=64, mode="resnet", sinkhorn_iters=5):
        super().__init__()
        self.num_layers = num_layers
        self.width = width
        self.mode = mode  # "resnet", "hc_naive", "mhc"
        self.sinkhorn_iters = sinkhorn_iters

        self.blocks = [nn.Linear(width, width) for _ in range(num_layers)]
        self.input_proj = nn.Linear(28 * 28, width)
        self.head = nn.Linear(width, 10)

        if self.mode in ["hc_naive", "mhc"]:
            self.mixing_logits = mx.random.normal((num_layers, num_layers)) * 0.01

    def _mixing_matrix(self):
        if self.mode == "mhc":
            return sinkhorn_knopp(self.mixing_logits, iters=self.sinkhorn_iters)
        if self.mode == "hc_naive":
            return mx.exp(self.mixing_logits)
        return None

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.input_proj(x))

        history = [x]
        mix_mat = self._mixing_matrix()
        current_state = x

        for i in range(self.num_layers):
            if self.mode == "resnet":
                inp = current_state
            else:
                weights = mix_mat[i, : len(history)]
                h_stack = mx.stack(history)
                inp = mx.tensordot(weights, h_stack, axes=([0], [0]))

            out = self.blocks[i](inp)
            out = nn.relu(out)

            if self.mode == "resnet":
                current_state = current_state + out
            else:
                current_state = out

            history.append(current_state)

        return self.head(current_state)
