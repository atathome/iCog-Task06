"""Task 4: Integrated serving pipeline with stress tests and benchmarks.

Components:
- Client threads
- Router
- Thread-safe LRU cache (coarse/scoped)
- DynamicBatchQueue
- Inference worker pool

Lock hierarchy:
1. Router stats lock (router._stats_lock) - short-lived counters only
2. Cache lock (inside LRUEmbeddingCache) - never held during embed/forward
3. Queue condition lock (inside DynamicBatchQueue)

The router does not hold cache and queue locks simultaneously. Cache and queue locks
are disjoint in scope. This prevents deadlocks.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any

from task2_lru_cache import LRUEmbeddingCache
from task3_batch_queue import DynamicBatchQueue, ShutdownError


@dataclass
class Request:
    request_id: int
    prompt: str
    submitted_at: float
    done: threading.Event
    result: Any = None
    error: str | None = None
    completed_at: float | None = None


class InferenceWorkerPool:
    def __init__(self, queue: DynamicBatchQueue[Request], num_workers: int, max_batch_size: int, batch_timeout_s: float, process_delay_s: float = 0.005):
        self.queue = queue
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_s
        self.process_delay_s = process_delay_s
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self.batches_processed = 0
        self.total_batch_items = 0
        self.total_forward_calls = 0

    def start(self) -> None:
        for i in range(self.num_workers):
            t = threading.Thread(target=self._run, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _run(self) -> None:
        while True:
            batch = self.queue.get_batch(self.max_batch_size, self.batch_timeout_s)
            if not batch:
                if self.queue.is_shutdown() and self.queue.qsize() == 0:
                    return
                continue
            # Simulate batched forward pass outside any lock.
            time.sleep(self.process_delay_s)
            with self._stats_lock:
                self.batches_processed += 1
                self.total_batch_items += len(batch)
                self.total_forward_calls += 1
            now = time.perf_counter()
            for req in batch:
                req.result = f"out:{req.prompt}"
                req.completed_at = now
                req.done.set()

    def join(self, timeout: float | None = None) -> None:
        for t in self._threads:
            t.join(timeout=timeout)

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            avg_batch = (self.total_batch_items / self.batches_processed) if self.batches_processed else 0.0
            util = (avg_batch / self.max_batch_size) if self.max_batch_size else 0.0
            return {
                "batches_processed": self.batches_processed,
                "total_batch_items": self.total_batch_items,
                "total_forward_calls": self.total_forward_calls,
                "avg_batch_size": avg_batch,
                "avg_batch_utilization": util,
            }


class Router:
    def __init__(self, cache: LRUEmbeddingCache, queue: DynamicBatchQueue[Request], embed_delay_s: float = 0.002):
        self.cache = cache
        self.queue = queue
        self.embed_delay_s = embed_delay_s
        self._stats_lock = threading.Lock()
        self.total_requests = 0
        self.accepted = 0
        self.completed = 0
        self.rejected_on_shutdown = 0
        self.errors = 0
        self.latencies_s: list[float] = []

    def _embed(self, prompt: str) -> str:
        time.sleep(self.embed_delay_s)
        return f"emb:{prompt}"

    def handle(self, req: Request) -> None:
        with self._stats_lock:
            self.total_requests += 1
            self.accepted += 1  # accepted by router; terminal outcome tracked below
        try:
            # Cache lookup/compute - scoped mode releases lock during embed.
            self.cache.get_or_compute(req.prompt, lambda p=req.prompt: self._embed(p))
            self.queue.put(req)
            # Wait for worker completion
            ok = req.done.wait(timeout=10.0)
            if not ok:
                req.error = "timeout"
                req.done.set()
                with self._stats_lock:
                    self.errors += 1
                    self.completed += 1  # terminal
                return
            with self._stats_lock:
                self.completed += 1
                if req.error:
                    self.errors += 1
                elif req.completed_at is not None:
                    self.latencies_s.append(req.completed_at - req.submitted_at)
        except ShutdownError:
            req.error = "rejected_on_shutdown"
            req.done.set()
            with self._stats_lock:
                self.rejected_on_shutdown += 1
        except Exception as e:
            req.error = repr(e)
            req.done.set()
            with self._stats_lock:
                self.errors += 1
                self.completed += 1  # terminal error after acceptance

    def snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            lats = list(self.latencies_s)
        avg_ms = (statistics.mean(lats) * 1000) if lats else 0.0
        p95_ms = 0.0
        if lats:
            ordered = sorted(lats)
            idx = max(0, int(0.95 * len(ordered)) - 1)
            p95_ms = ordered[idx] * 1000
        with self._stats_lock:
            return {
                "total_requests": self.total_requests,
                "accepted": self.accepted,
                "completed": self.completed,
                "rejected_on_shutdown": self.rejected_on_shutdown,
                "errors": self.errors,
                "avg_latency_ms": avg_ms,
                "p95_latency_ms": p95_ms,
            }


def run_pipeline(
    *,
    num_clients: int = 30,
    requests_per_client: int = 20,
    cache_capacity: int = 32,
    cache_mode: str = "scoped",
    queue_capacity: int = 64,
    num_workers: int = 4,
    max_batch_size: int = 8,
    batch_timeout_s: float = 0.02,
    embed_delay_s: float = 0.002,
    forward_delay_s: float = 0.005,
    shutdown_after_s: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    random.seed(seed)
    cache = LRUEmbeddingCache(capacity=cache_capacity, mode=cache_mode)
    queue = DynamicBatchQueue[Request](capacity=queue_capacity)
    router = Router(cache=cache, queue=queue, embed_delay_s=embed_delay_s)
    pool = InferenceWorkerPool(queue, num_workers=num_workers, max_batch_size=max_batch_size, batch_timeout_s=batch_timeout_s, process_delay_s=forward_delay_s)
    pool.start()

    prompts = [f"prompt-{i}" for i in range(40)]  # mixed hit/miss; repeated
    req_counter = iter(range(num_clients * requests_per_client + 1000))
    start_barrier = threading.Barrier(num_clients)
    t0 = time.perf_counter()

    stop_event = threading.Event()
    if shutdown_after_s is not None:
        def _shutdown_trigger():
            time.sleep(shutdown_after_s)
            queue.shutdown()
            stop_event.set()
        threading.Thread(target=_shutdown_trigger, daemon=True, name="shutdown-trigger").start()

    def client(cid: int):
        rng = random.Random(seed + cid)
        try:
            start_barrier.wait(timeout=10)
        except Exception:
            return
        for _ in range(requests_per_client):
            rid = next(req_counter)
            # Bursty arrivals: occasional pause
            if rng.random() < 0.15:
                time.sleep(rng.uniform(0.0, 0.01))
            prompt = rng.choice(prompts[:20]) if rng.random() < 0.7 else rng.choice(prompts)
            req = Request(request_id=rid, prompt=prompt, submitted_at=time.perf_counter(), done=threading.Event())
            router.handle(req)
            time.sleep(rng.uniform(0.0, 0.003))

    threads = [threading.Thread(target=client, args=(i,), name=f"client-{i}") for i in range(num_clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Drain: shutdown if not already
    queue.shutdown()
    pool.join(timeout=10)
    elapsed = time.perf_counter() - t0

    cache_stats = cache.snapshot_stats()
    total_lookups = cache_stats["hits"] + cache_stats["misses"]
    hit_rate = (cache_stats["hits"] / total_lookups) if total_lookups else 0.0
    router_stats = router.snapshot()
    pool_stats = pool.stats()
    ok_integrity, integrity_msg = cache.verify_integrity()

    result = {
        "config": {
            "num_clients": num_clients,
            "requests_per_client": requests_per_client,
            "cache_capacity": cache_capacity,
            "cache_mode": cache_mode,
            "queue_capacity": queue_capacity,
            "num_workers": num_workers,
            "max_batch_size": max_batch_size,
            "batch_timeout_s": batch_timeout_s,
            "embed_delay_s": embed_delay_s,
            "forward_delay_s": forward_delay_s,
            "shutdown_after_s": shutdown_after_s,
        },
        "elapsed_s": elapsed,
        "throughput_rps": (router_stats["completed"] / elapsed) if elapsed else 0.0,
        "router": router_stats,
        "cache": {**cache_stats, "hit_rate": hit_rate},
        "queue": queue.stats(),
        "worker": pool_stats,
        "integrity": {"ok": ok_integrity, "msg": integrity_msg},
        "checks": {
            "accepted_eq_completed_plus_rejected": router_stats["accepted"] == router_stats["completed"] + router_stats["rejected_on_shutdown"],
            "no_deadlock_join": all(not t.is_alive() for t in threads),
        },
    }
    return result


def benchmark() -> dict[str, Any]:
    # Keep runtimes reasonable while showing trade-offs
    base = dict(
        num_clients=30,
        requests_per_client=25,
        cache_capacity=32,
        queue_capacity=128,
        num_workers=4,
        embed_delay_s=0.003,
        forward_delay_s=0.006,
        seed=1,
    )

    configs = [
        ("coarse_unbatched", {**base, "cache_mode": "coarse", "max_batch_size": 1, "batch_timeout_s": 0.005}),
        ("coarse_batched",   {**base, "cache_mode": "coarse", "max_batch_size": 8, "batch_timeout_s": 0.02}),
        ("scoped_unbatched", {**base, "cache_mode": "scoped", "max_batch_size": 1, "batch_timeout_s": 0.005}),
        ("scoped_batched",   {**base, "cache_mode": "scoped", "max_batch_size": 8, "batch_timeout_s": 0.02}),
    ]

    results = {}
    for name, cfg in configs:
        print(f"Running {name}...", flush=True)
        res = run_pipeline(**cfg)
        results[name] = res
    return results


def format_comparison_table(results: dict[str, Any]) -> str:
    headers = ["config", "rps", "avg_ms", "p95_ms", "hit_rate", "avg_batch", "batch_util", "accepted==comp+rej", "integrity"]
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            f"{r['throughput_rps']:.2f}",
            f"{r['router']['avg_latency_ms']:.2f}",
            f"{r['router']['p95_latency_ms']:.2f}",
            f"{r['cache']['hit_rate']:.3f}",
            f"{r['worker']['avg_batch_size']:.2f}",
            f"{r['worker']['avg_batch_utilization']:.3f}",
            str(r['checks']['accepted_eq_completed_plus_rejected']),
            str(r['integrity']['ok']),
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    def fmt(row):
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * w for w in widths)
    out = [fmt(headers), sep]
    out.extend(fmt(r) for r in rows)
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stress", "benchmark"], default="stress")
    args = parser.parse_args()
    if args.mode == "stress":
        res = run_pipeline(num_clients=30, requests_per_client=20, cache_mode="scoped", max_batch_size=8, batch_timeout_s=0.02, shutdown_after_s=0.5, seed=2)
        print(json.dumps(res, indent=2))
    else:
        res = benchmark()
        print(format_comparison_table(res))
        print("\nRaw JSON saved to stdout not shown; use programmatic call for full data.")
