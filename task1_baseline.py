"""Task 1: Broken baseline inference router.

Intentionally unsynchronized shared state to expose concurrency bugs.
NOT for production use.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field


@dataclass
class BaselineMetrics:
    total_requests: int = 0
    cache_hits: int = 0
    total_tokens: int = 0
    duplicate_embed_calls: int = 0
    queue_overflows: int = 0
    queue_dropped: int = 0


class BrokenInferenceRouter:
    """Unsynchronized baseline with deliberate race windows."""

    def __init__(self, queue_capacity: int = 8):
        self.cache: dict[str, str] = {}  # unsynchronized dict
        self.queue: list[str] = []  # plain list queue
        self.queue_capacity = queue_capacity
        self.metrics = BaselineMetrics()
        self._embed_call_log: list[str] = []
        self._log_lock = threading.Lock()  # only for observation logging, not protecting shared state

    def embed(self, prompt: str) -> str:
        # Simulated embedding/inference work
        with self._log_lock:
            self._embed_call_log.append(prompt)
        time.sleep(random.uniform(0.001, 0.01))
        return f"vec({prompt})"

    def handle_request(self, prompt: str) -> str:
        # RACE 1: read-modify-write on total_requests (no lock)
        current = self.metrics.total_requests
        time.sleep(random.uniform(0.0, 0.002))
        self.metrics.total_requests = current + 1

        # RACE 2: cache stampede - check-then-act on unsynchronized dict
        if prompt in self.cache:
            # widen race window
            time.sleep(random.uniform(0.0, 0.002))
            self.metrics.cache_hits += 1  # unsynchronized
            embedding = self.cache[prompt]
        else:
            # Multiple threads can miss simultaneously and compute same key.
            embedding = self.embed(prompt)
            # Another race window before storing
            time.sleep(random.uniform(0.0, 0.002))
            if prompt in self.cache:
                # Detected duplicate work after the fact
                self.metrics.duplicate_embed_calls += 1
            self.cache[prompt] = embedding

        # RACE 3: plain list queue with capacity check-then-append
        if len(self.queue) >= self.queue_capacity:
            self.metrics.queue_overflows += 1
            self.metrics.queue_dropped += 1
        else:
            time.sleep(random.uniform(0.0, 0.002))  # widen window
            self.queue.append(prompt)
            # If multiple threads pass capacity check, list grows beyond capacity.

        # RACE 4: read-modify-write on total_tokens
        tokens = len(prompt.split())
        tok_cur = self.metrics.total_tokens
        time.sleep(random.uniform(0.0, 0.001))
        self.metrics.total_tokens = tok_cur + tokens

        # Simulate dequeue/processing without synchronization (can race with appends)
        time.sleep(random.uniform(0.0, 0.002))
        if self.queue:
            try:
                self.queue.pop(0)
            except (IndexError, RuntimeError):
                pass
        return embedding


def run_baseline_stress(num_clients: int = 25, requests_per_client: int = 20, seed: int = 42) -> dict:
    random.seed(seed)
    router = BrokenInferenceRouter(queue_capacity=8)
    prompts = [f"prompt-{i % 6}" for i in range(30)]  # small key set to force stampedes
    expected_requests = num_clients * requests_per_client
    expected_tokens = 0
    expected_lock = threading.Lock()

    errors: list[str] = []
    err_lock = threading.Lock()
    start_barrier = threading.Barrier(num_clients)

    def client(cid: int):
        local_random = random.Random(seed + cid)
        try:
            start_barrier.wait(timeout=5)
        except Exception as e:
            with err_lock:
                errors.append(f"barrier:{e}")
            return
        for _ in range(requests_per_client):
            p = local_random.choice(prompts)
            with expected_lock:
                nonlocal expected_tokens
                expected_tokens += len(p.split())
            try:
                router.handle_request(p)
            except Exception as e:
                with err_lock:
                    errors.append(repr(e))
            time.sleep(local_random.uniform(0.0, 0.003))

    threads = [threading.Thread(target=client, args=(i,), name=f"client-{i}") for i in range(num_clients)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    # Observe symptoms
    with router._log_lock:
        embed_calls = list(router._embed_call_log)
    embed_counts: dict[str, int] = {}
    for p in embed_calls:
        embed_counts[p] = embed_counts.get(p, 0) + 1
    duplicate_keys = {k: v for k, v in embed_counts.items() if v > 1}

    summary = {
        "expected_requests": expected_requests,
        "observed_total_requests": router.metrics.total_requests,
        "lost_request_updates": expected_requests - router.metrics.total_requests,
        "expected_total_tokens": expected_tokens,
        "observed_total_tokens": router.metrics.total_tokens,
        "lost_token_updates": expected_tokens - router.metrics.total_tokens,
        "cache_hits_counter": router.metrics.cache_hits,
        "duplicate_embed_detected_after_store": router.metrics.duplicate_embed_calls,
        "embed_calls_per_key": embed_counts,
        "keys_with_duplicate_embed_calls": duplicate_keys,
        "queue_capacity": router.queue_capacity,
        "queue_len_at_end": len(router.queue),
        "queue_overflow_counter": router.metrics.queue_overflows,
        "queue_dropped_counter": router.metrics.queue_dropped,
        "errors": errors,
        "elapsed_s": elapsed,
    }
    return summary


if __name__ == "__main__":
    import json

    result = run_baseline_stress()
    print(json.dumps(result, indent=2))
