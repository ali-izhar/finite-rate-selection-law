# Models

The trained Boltzmann machine of the MNIST illustration ([docs/mnist-illustration.md](../docs/mnist-illustration.md)).
No paper result uses these files. `src/dbm.py` loads both.

| File | Content |
|---|---|
| `boltzmann_machine.npz` | Two-layer Boltzmann machine on binarized MNIST: `W1` (784 x 500), `W2` (500 x 500), biases `b`, `c1`, `c2`, their centred forms, and a training history. The training script is not recorded; the original file name recorded 600 epochs and learning rate 0.002. |
| `digit_classifier.npz` | Linear classifier on the 784 pixels (`coef` 10 x 784, `intercept` 10) that labels the final samples. Its training is not recorded. |

Commands: [docs/reproduce.md](../docs/reproduce.md).
