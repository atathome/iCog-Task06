"""Tinygrad model used by the inference server.

Simple text classifier: token ids -> embedding table -> mean pool -> linear logits.
The embedding stage is cacheable by prompt.
"""
from __future__ import annotations

import os
# Portable default for this environment; training script can override if needed.
os.environ.setdefault("DEV", "PYTHON")

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

import numpy as np
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_state_dict, load_state_dict

VOCAB_SIZE = 128
EMBED_DIM = 32
NUM_CLASSES = 8
MAX_LEN = 16
MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
WEIGHTS_PATH = MODEL_DIR / "tiny_embed_classifier.safetensors"
META_PATH = MODEL_DIR / "tiny_embed_classifier.json"


class TinyEmbedClassifier:
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM, num_classes: int = NUM_CLASSES):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        # token_ids: [B, L]
        emb = self.embedding(token_ids)  # [B, L, D]
        return emb.mean(axis=1)  # [B, D]

    def logits_from_embedding(self, emb: Tensor) -> Tensor:
        return self.fc(emb)

    def __call__(self, token_ids: Tensor) -> Tensor:
        return self.logits_from_embedding(self.embed_tokens(token_ids))


def tokenize(prompt: str, max_len: int = MAX_LEN, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    # Deterministic lightweight tokenizer: bytes -> vocab ids, padded/truncated.
    data = prompt.encode("utf-8", errors="ignore")[:max_len]
    arr = np.zeros((max_len,), dtype=np.int32)
    for i, b in enumerate(data):
        # keep 0 for padding; map bytes into 1..vocab_size-1
        arr[i] = 1 + (int(b) % (vocab_size - 1))
    return arr


def prompt_label(prompt: str, num_classes: int = NUM_CLASSES) -> int:
    # Synthetic label function used for training data; deterministic.
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int(h[0]) % num_classes


@dataclass
class LoadedModel:
    model: TinyEmbedClassifier
    meta: dict


def save_model(model: TinyEmbedClassifier, path: Path = WEIGHTS_PATH, meta_path: Path = META_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().realize().numpy() for k, v in get_state_dict(model).items()}
    np.savez(str(path.with_suffix(".npz")), **state)
    meta = {
        "vocab_size": model.vocab_size,
        "embed_dim": model.embed_dim,
        "num_classes": model.num_classes,
        "max_len": MAX_LEN,
        "weights": path.with_suffix(".npz").name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def load_model(path: Path = WEIGHTS_PATH, meta_path: Path = META_PATH) -> LoadedModel:
    meta = json.loads(meta_path.read_text())
    model = TinyEmbedClassifier(meta["vocab_size"], meta["embed_dim"], meta["num_classes"])
    weights_path = path.with_suffix(".npz")
    if not weights_path.exists():
        weights_path = path.parent / meta.get("weights", weights_path.name)
    npz = np.load(str(weights_path), allow_pickle=False)
    state = {k: Tensor(npz[k]) for k in npz.files}
    load_state_dict(model, state)
    # Realize loaded params
    for p in get_state_dict(model).values():
        p.realize()
    return LoadedModel(model=model, meta=meta)


def embed_prompt(model: TinyEmbedClassifier, prompt: str) -> np.ndarray:
    toks = tokenize(prompt, max_len=MAX_LEN, vocab_size=model.vocab_size)[None, :]
    emb = model.embed_tokens(Tensor(toks)).realize()
    return emb.numpy().astype(np.float32)[0]


def infer_with_cached_embedding(model: TinyEmbedClassifier, prompt: str, emb_vec: np.ndarray) -> dict:
    emb = Tensor(emb_vec[None, :].astype(np.float32))
    logits = model.logits_from_embedding(emb).realize()
    probs = logits.softmax(axis=-1).numpy()[0]
    pred = int(probs.argmax())
    return {"pred": pred, "confidence": float(probs[pred])}
