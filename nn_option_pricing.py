"""
Neural network from scratch (NumPy only) that learns to price a European call option.

The target is the Black-Scholes price, so the true answer is known and the network's
error can be measured exactly. After training, the script draws the loss landscape
around the solution and the path gradient descent took through it.

Usage:
    python nn_option_pricing.py

Outputs (in ./output):
    results.png         training curves, predicted vs true price, error by moneyness
    loss_landscape.png  loss surface in the plane of the two main directions of the
                        training path, with the path drawn on top
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
N_TRAIN = 20_000
N_TEST = 5_000
HIDDEN = (32, 32)  # two hidden layers
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 0.05
MOMENTUM = 0.9
GRID_POINTS = 41  # resolution of the loss landscape
OUT_DIR = Path("output")

# Input ranges: moneyness S/K, maturity (years), volatility, interest rate
RANGES = {
    "moneyness": (0.7, 1.3),
    "maturity": (0.1, 2.0),
    "volatility": (0.1, 0.5),
    "rate": (0.0, 0.05),
}


# --------------------------------------------------------------------------
# Data: Black-Scholes call price (strike normalised to 1)
# --------------------------------------------------------------------------
_erf = np.vectorize(math.erf)


def norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def black_scholes_call(m, T, sigma, r):
    """Call price per unit of strike, with m = S / K."""
    d1 = (np.log(m) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return m * norm_cdf(d1) - np.exp(-r * T) * norm_cdf(d2)


def make_data(n, rng):
    cols = [rng.uniform(*RANGES[k], size=n) for k in RANGES]
    X = np.column_stack(cols)
    y = black_scholes_call(*cols)[:, None]
    return X, y


# --------------------------------------------------------------------------
# Network: fully connected, tanh activations, linear output, MSE loss
# --------------------------------------------------------------------------
def init_params(sizes, rng):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_in, fan_out))
        b = np.zeros(fan_out)
        params += [W, b]
    return params


def forward(params, X):
    """Return the output and the activations needed for backprop."""
    activations = [X]
    a = X
    n_layers = len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        z = a @ W + b
        a = z if i == n_layers - 1 else np.tanh(z)
        activations.append(a)
    return a, activations


def mse(params, X, y):
    pred, _ = forward(params, X)
    return float(np.mean((pred - y) ** 2))


def gradients(params, X, y):
    """Backpropagation, written out by hand."""
    pred, acts = forward(params, X)
    n = X.shape[0]
    delta = 2.0 * (pred - y) / n  # d loss / d output
    grads = [None] * len(params)
    n_layers = len(params) // 2
    for i in reversed(range(n_layers)):
        W = params[2 * i]
        a_prev = acts[i]
        grads[2 * i] = a_prev.T @ delta
        grads[2 * i + 1] = delta.sum(axis=0)
        if i > 0:
            delta = (delta @ W.T) * (1.0 - acts[i] ** 2)  # tanh derivative
    return grads


def flatten(params):
    return np.concatenate([p.ravel() for p in params])


def unflatten(vec, template):
    out, start = [], 0
    for p in template:
        out.append(vec[start : start + p.size].reshape(p.shape))
        start += p.size
    return out


def gradient_check(params, X, y, rng, n_checks=8, eps=1e-6):
    """Compare backprop with finite differences on a few random weights."""
    analytic = flatten(gradients(params, X, y))
    base = flatten(params)
    worst = 0.0
    for idx in rng.choice(base.size, size=n_checks, replace=False):
        up, down = base.copy(), base.copy()
        up[idx] += eps
        down[idx] -= eps
        numeric = (mse(unflatten(up, params), X, y) - mse(unflatten(down, params), X, y)) / (2 * eps)
        denom = max(abs(numeric) + abs(analytic[idx]), 1e-12)
        worst = max(worst, abs(numeric - analytic[idx]) / denom)
    return worst


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(params, X_tr, y_tr, X_te, y_te, rng):
    velocity = [np.zeros_like(p) for p in params]
    history = {"train": [], "test": []}
    snapshots = [flatten(params)]

    for epoch in range(EPOCHS):
        order = rng.permutation(len(X_tr))
        for start in range(0, len(X_tr), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            grads = gradients(params, X_tr[idx], y_tr[idx])
            for p, g, v in zip(params, grads, velocity):
                v *= MOMENTUM
                v -= LEARNING_RATE * g
                p += v
        history["train"].append(mse(params, X_tr, y_tr))
        history["test"].append(mse(params, X_te, y_te))
        snapshots.append(flatten(params))
        if (epoch + 1) % 25 == 0:
            print(f"epoch {epoch + 1:3d}  train MSE {history['train'][-1]:.2e}  "
                  f"test MSE {history['test'][-1]:.2e}")
    return history, np.array(snapshots)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_results(history, params, X_te_raw, X_te, y_te, path):
    pred, _ = forward(params, X_te)
    err = (pred - y_te).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("Neural network (NumPy only) pricing a European call, target = Black-Scholes",
                 fontweight="bold")

    ax = axes[0]
    ax.semilogy(history["train"], label="Train")
    ax.semilogy(history["test"], label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (price per unit of strike)")
    ax.set_title("Training curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(y_te, pred, s=3, alpha=0.3)
    lim = [0, float(max(y_te.max(), pred.max()))]
    ax.plot(lim, lim, color="black", lw=1)
    ax.set_xlabel("Black-Scholes price")
    ax.set_ylabel("Network price")
    ax.set_title("Test set: predicted vs true")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(X_te_raw[:, 0], err, s=3, alpha=0.3)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Moneyness S/K")
    ax.set_ylabel("Error (network - true)")
    ax.set_title("Error by moneyness")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(np.sqrt(np.mean(err**2))), float(np.abs(err).max())


def plot_landscape(snapshots, template, X, y, path):
    """Loss surface in the 2D plane spanned by the main directions of the training path."""
    final = snapshots[-1]
    deltas = snapshots - final
    # Principal directions of the path (SVD of the centred trajectory)
    _, s, vt = np.linalg.svd(deltas, full_matrices=False)
    v1, v2 = vt[0], vt[1]
    explained = s**2 / np.sum(s**2)
    coords = np.column_stack([deltas @ v1, deltas @ v2])

    pad = 0.25
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = hi - lo
    a = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], GRID_POINTS)
    b = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], GRID_POINTS)

    Z = np.zeros((GRID_POINTS, GRID_POINTS))
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            Z[j, i] = mse(unflatten(final + ai * v1 + bj * v2, template), X, y)
    logZ = np.log10(Z)
    A, B = np.meshgrid(a, b)

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("Loss landscape and the path taken by gradient descent", fontweight="bold")

    ax = fig.add_subplot(1, 2, 1)
    cs = ax.contourf(A, B, logZ, levels=25, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10(MSE)")
    ax.plot(coords[:, 0], coords[:, 1], "-o", color="white", ms=3, lw=1.2)
    ax.plot(*coords[0], "o", color="red", ms=8, label="Start")
    ax.plot(*coords[-1], "*", color="gold", ms=14, label="End")
    ax.set_xlabel(f"Direction 1 ({explained[0]:.0%} of path variance)")
    ax.set_ylabel(f"Direction 2 ({explained[1]:.0%} of path variance)")
    ax.set_title("Contour view")
    ax.legend(frameon=True)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(A, B, logZ, cmap="viridis", alpha=0.75, linewidth=0)
    path_loss = np.array([np.log10(mse(unflatten(final + c[0] * v1 + c[1] * v2, template), X, y))
                          for c in coords])
    ax.plot(coords[:, 0], coords[:, 1], path_loss, color="white", lw=2)
    ax.plot(*coords[0], path_loss[0], "o", color="red", ms=6)
    ax.plot(*coords[-1], path_loss[-1], "*", color="gold", ms=10)
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_zlabel("log10(MSE)")
    ax.set_title("3D view")
    ax.view_init(elev=35, azim=-60)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return explained[:2]


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    X_tr_raw, y_tr = make_data(N_TRAIN, rng)
    X_te_raw, y_te = make_data(N_TEST, rng)

    mean, std = X_tr_raw.mean(axis=0), X_tr_raw.std(axis=0)  # scale inputs with train stats
    X_tr, X_te = (X_tr_raw - mean) / std, (X_te_raw - mean) / std

    params = init_params([X_tr.shape[1], *HIDDEN, 1], rng)
    n_params = sum(p.size for p in params)
    print(f"Network: 4 -> {' -> '.join(map(str, HIDDEN))} -> 1  ({n_params} parameters)")

    err = gradient_check(params, X_tr[:64], y_tr[:64], rng)
    print(f"Gradient check, worst relative error vs finite differences: {err:.2e}")

    history, snapshots = train(params, X_tr, y_tr, X_te, y_te, rng)

    OUT_DIR.mkdir(exist_ok=True)
    rmse, max_err = plot_results(history, params, X_te_raw, X_te, y_te, OUT_DIR / "results.png")
    print(f"\nTest RMSE: {rmse:.5f}   Max abs error: {max_err:.5f}   (price per unit of strike)")

    sub = rng.choice(N_TRAIN, size=2000, replace=False)  # subset keeps the landscape fast
    explained = plot_landscape(snapshots, params, X_tr[sub], y_tr[sub], OUT_DIR / "loss_landscape.png")
    print(f"Landscape plane explains {explained.sum():.0%} of the path variance")
    print(f"Saved figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
