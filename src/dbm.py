"""Block-Gibbs sampler for the two-layer Boltzmann machine trained on MNIST.

``models/boltzmann_machine.npz`` holds the weights and biases in the
standard (un-centred) parametrisation, ``DBM(W1, W2, b, c1, c2)``.

Energy at inverse temperature beta:
    E(v, h1, h2) = -(v @ W1 @ h1) - (h1 @ W2 @ h2) - b.v - c1.h1 - c2.h2
    p(state) ~ exp(-beta E)
One ``step`` is one block-Gibbs sweep: h2 | h1, then h1 | v, h2, then v | h1.
This re-implements the sampler that made the saved ensemble
(results/experiments/dbm_reheat.npz); its update order and random stream
differ, so a rerun reproduces the ensemble only statistically.

The raw MNIST IDX files are not in the repository.  Set MNIST_RAW to their
directory; the default is ~/.cache/mnist_data/MNIST/raw.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MNIST_RAW = os.environ.get("MNIST_RAW", str(Path.home() / ".cache/mnist_data/MNIST/raw"))  # raw IDX files
MODEL = ROOT / "models" / "boltzmann_machine.npz"
PROBE = ROOT / "models" / "digit_classifier.npz"


def _idx(path):
    """Read one MNIST IDX file (images as (n, 784) uint8, labels as (n,) uint8)."""
    with open(path, "rb") as f:
        data = f.read()
    magic, n = struct.unpack(">II", data[:8])
    if magic == 2051:
        r, c = struct.unpack(">II", data[8:16])
        return np.frombuffer(data[16:], np.uint8).reshape(n, r * c)
    return np.frombuffer(data[8:], np.uint8)


def load_mnist_binary(raw=MNIST_RAW):
    """MNIST training set, pixels scaled by 1/255 and thresholded at 0.5.

    Returns ``(Xb, y)``: binary images (60000, 784) float64 and integer labels.
    Draws no random numbers.
    """
    X = _idx(f"{raw}/train-images-idx3-ubyte").astype(np.float64) / 255.0
    y = _idx(f"{raw}/train-labels-idx1-ubyte").astype(int)
    return (X > 0.5).astype(np.float64), y


def load_probe(path=PROBE):
    """The linear class probe: returns ``(coef, intercept)``; logits are ``v @ coef.T + intercept``."""
    cl = np.load(path)
    return cl["coef"], cl["intercept"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class DBM:
    def __init__(self, W1, W2, b, c1, c2):
        self.W1, self.W2 = np.asarray(W1, float), np.asarray(W2, float)
        self.b, self.c1, self.c2 = np.asarray(b, float), np.asarray(c1, float), np.asarray(c2, float)

    def _init_hidden(self, v, beta, rng):
        """Bottom-up initialisation: h1 | v (with h2 = 0), then h2 | h1."""
        h1 = (rng.random(v.shape[:-1] + (self.W1.shape[1],)) < _sigmoid(beta * (self.c1 + v @ self.W1))).astype(float)
        h2 = (rng.random(v.shape[:-1] + (self.W2.shape[1],)) < _sigmoid(beta * (self.c2 + h1 @ self.W2))).astype(float)
        return h1, h2

    def step(self, v, h1, h2, beta, rng):
        h2 = (rng.random(h2.shape) < _sigmoid(beta * (self.c2 + h1 @ self.W2))).astype(float)
        h1 = (rng.random(h1.shape) < _sigmoid(beta * (self.c1 + v @ self.W1 + h2 @ self.W2.T))).astype(float)
        v = (rng.random(v.shape) < _sigmoid(beta * (self.b + h1 @ self.W1.T))).astype(float)
        return v, h1, h2
