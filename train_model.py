"""Train and save the tinygrad embedding classifier used by benchmarks.

Run once to produce artifacts/tiny_embed_classifier.safetensors.
Benchmarks/stress tests should only load the saved model.
"""
from __future__ import annotations

import os
# Use PYTHON backend for portability in this environment (CUDA triggers CPU clang compilation path).
os.environ.setdefault("DEV", "PYTHON")

import json
import random
from pathlib import Path

import numpy as np
from tinygrad import Tensor, nn
from tinygrad.helpers import Context
from tinygrad.nn.state import get_parameters

from ml_model import (
    MAX_LEN,
    META_PATH,
    MODEL_DIR,
    NUM_CLASSES,
    VOCAB_SIZE,
    WEIGHTS_PATH,
    TinyEmbedClassifier,
    prompt_label,
    save_model,
    tokenize,
)


def make_dataset(n: int = 512, seed: int = 0):
    rng = random.Random(seed)
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "fox", "gpu", "cache", "batch",
        "router", "queue", "latency", "throughput", "model", "embed", "token", "server",
        "client", "worker", "lock", "scope", "hit", "miss", "evict"
    ]
    prompts = []
    labels = []
    for _ in range(n):
        k = rng.randint(1, 4)
        prompt = " ".join(rng.choice(words) for _ in range(k))
        prompts.append(prompt)
        labels.append(prompt_label(prompt, NUM_CLASSES))
    x = np.stack([tokenize(p, MAX_LEN, VOCAB_SIZE) for p in prompts], axis=0).astype(np.int32)
    y = np.array(labels, dtype=np.int32)
    return x, y


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    pred = logits.argmax(axis=-1)
    return float((pred == y).mean())


def main():
    random.seed(0)
    np.random.seed(0)
    Tensor.manual_seed(0)

    x_train, y_train = make_dataset(32, seed=1)
    x_val, y_val = make_dataset(16, seed=2)
    model = TinyEmbedClassifier()
    opt = nn.optim.Adam(get_parameters(model), lr=1e-2)

    batch_size = 8
    steps = 5
    losses = []
    with Context(TRAINING=1):
        for step in range(steps):
            idx = np.random.randint(0, len(x_train), size=batch_size)
            xb = Tensor(x_train[idx])
            yb = y_train[idx]
            logits = model(xb)
            loss = logits.sparse_categorical_crossentropy(Tensor(yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.realize().numpy())
            losses.append(loss_val)
            if step % 50 == 0 or step == steps - 1:
                with Context(TRAINING=0):
                    val_logits = model(Tensor(x_val)).realize().numpy()
                acc = accuracy(val_logits, y_val)
                print(f"step={step:3d} loss={loss_val:.4f} val_acc={acc:.3f}")

    with Context(TRAINING=0):
        val_logits = model(Tensor(x_val)).realize().numpy()
    val_acc = accuracy(val_logits, y_val)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model)
    metrics = {
        "steps": steps,
        "batch_size": batch_size,
        "final_loss": float(losses[-1]),
        "val_acc": val_acc,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "backend": os.environ.get("DEV", ""),
    }
    (MODEL_DIR / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved:", WEIGHTS_PATH.with_suffix('.npz'))
    print("Metrics:", json.dumps(metrics))


if __name__ == "__main__":
    main()
