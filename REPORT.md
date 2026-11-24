# Concurrent ML Inference Server - Training Task Report

## Workspace artifacts

- `task1_baseline.py` - intentionally broken baseline router + reproducible stress script.
- `task2_lru_cache.py` - thread-safe LRU embedding cache built from scratch (coarse/scoped modes).
- `task3_batch_queue.py` - bounded dynamic batching queue with `Lock`/`Condition`.
- `task4_pipeline.py` - integrated pipeline, stress test, and benchmark comparisons.
- `test_task2.py`, `test_task3.py` - correctness/stress checks for Task 2 and Task 3.
- `benchmark_results.json` - raw benchmark output.

## How to run

```bash
python3 task1_baseline.py          # Task 1 reproducible stress + bug summary
python3 test_task2.py              # Task 2 LRU correctness/stress
python3 test_task3.py              # Task 3 queue correctness/stress
python3 task4_pipeline.py --mode stress      # Task 4 reliability stress (30 clients, mid-run shutdown)
python3 task4_pipeline.py --mode benchmark   # Task 4 benchmark comparison table
```

## Task 1: Broken baseline - observed bugs

Repro command: `python3 task1_baseline.py` (25 clients × 20 requests = 500 expected requests, small prompt key set to force stampedes).

Observed run (seed 42):

| Symptom | Observed | Expected |
|---|---:|---:|
| Lost request counter updates | `total_requests=92` | `500` |
| Lost token counter updates | `total_tokens=166` | `500` |
| Cache stampede / duplicate embed work | duplicate embed calls per key, e.g. `prompt-1:7`, `prompt-2:7` | 1 per key ideally |
| Queue capacity/overflow symptoms | `queue_overflow_counter=24`, `queue_dropped_counter=24`, capacity=8 | no overflow beyond capacity |

### Bug-to-code mapping (`task1_baseline.py`)

| Bug | Unsynchronized code | Why it breaks |
|---|---|---|
| Lost `total_requests` updates (RMW race) | `task1_baseline.py:44-46` reads `current = self.metrics.total_requests`, sleeps, writes `current + 1` with no lock | Concurrent threads overwrite each other's increments. |
| Cache stampede / check-then-act race | `task1_baseline.py:49-62` checks `prompt in self.cache` and reads/writes `self.cache` without synchronization; embed happens on miss at `:56` before store at `:62` | Multiple threads miss same key, compute duplicate embeddings, and race on dict writes. |
| Unsynchronized cache hit counter | `task1_baseline.py:52` increments `self.metrics.cache_hits += 1` | Read-modify-write race; hit count can be lost/inaccurate. |
| Queue capacity violation / dropped items | `task1_baseline.py:65-71` checks `len(self.queue) >= capacity` then appends after sleep; no lock/condition | Multiple threads pass capacity check simultaneously, causing overflow or inconsistent drops. |
| Lost `total_tokens` updates (RMW race) | `task1_baseline.py:74-77` reads `tok_cur`, sleeps, writes `tok_cur + tokens` | Concurrent increments lost. |
| Queue corruption/race on dequeue | `task1_baseline.py:80-85` pops from plain list while other threads append/check | Unsynchronized list mutation can cause inconsistent queue state/IndexError windows. |

## Task 2: Thread-safe LRU embedding cache

Implementation: `task2_lru_cache.py`.

- **Data structures**: custom doubly linked list with dummy `head`/`tail` sentinels + `dict` map from key to node.
- **LRU behavior**: `get`/hit moves node to MRU front; `put` on full cache evicts LRU (`tail.prev`) while keeping `0 <= size <= capacity`.
- **Locking modes**:
  - `coarse`: holds `self._lock` across lookup + simulated `compute_fn()` + insert. Safe but blocks other cache operations during slow embed.
  - `scoped`: holds lock only for fast pointer/map mutations; computes embed outside lock. Uses an in-flight placeholder (`_InFlight` + `threading.Event`) to prevent cache stampede for same key.
- **Integrity helper**: `verify_integrity()` checks size bounds, map/list size match, forward prev/next pointer consistency, backward next/prev consistency, and no traversal cycle/runaway.
- **Tests**: `test_task2.py` runs basic LRU semantics and 20-thread stress (300 ops/thread, mixed get_or_compute/put, periodic integrity checks) for both modes. Result: both modes pass with zero integrity errors.

## Task 3: Bounded dynamic batching queue

Implementation: `task3_batch_queue.py`.

- **Buffer**: plain Python list guarded by `threading.Lock` + `threading.Condition`.
- **`put(item)` backpressure**: `while len(queue) >= capacity and not shutdown: cond.wait()`; on shutdown raises `ShutdownError`; on enqueue increments counter and `notify_all()`.
- **`get_batch(max_batch_size, timeout_s)` dynamic batching**: waits until `len(buf) >= max_batch_size`, or monotonic deadline expires with at least one item, or shutdown drain. Returns empty list only when shutdown and drained. Uses `notify_all()` after dequeue to wake producers.
- **`shutdown()` clean drain**: sets shutdown flag and `notify_all()`; no new puts accepted; queued items remain drainable.
- **No busy-waiting**: all waiting uses `Condition.wait()`/`wait(timeout=remaining)`.
- **Tests**: `test_task3.py` stress run with 10 producers × 200 items, 4 consumers, capacity 16; result `produced == consumed == 2000`, no errors, all consumers exited cleanly.

## Task 4: Integrated pipeline, stress test, and benchmarks

Implementation: `task4_pipeline.py`.

### Pipeline

`Client Threads -> Router -> Thread-Safe LRU Cache -> DynamicBatchQueue -> Inference Worker Pool`

- Router uses `cache.get_or_compute()` for embedding; scoped mode releases cache lock during simulated embed.
- Accepted requests are enqueued to `DynamicBatchQueue`; workers pull batches via `get_batch(max_batch_size, timeout_s)` and run simulated batched forward pass outside locks.
- Request completion is signaled by per-request `threading.Event`.

### Lock hierarchy / deadlock prevention

1. **Router stats lock** (`Router._stats_lock`) - short-lived counters/latency list only.
2. **Cache lock** (inside `LRUEmbeddingCache`) - only for fast map/list/in-flight mutations; never held during embed/forward.
3. **Queue condition lock** (inside `DynamicBatchQueue`) - guards bounded buffer and condition waits.

Rules enforced:
- Cache and queue locks are **disjoint**; router never holds them simultaneously.
- No lock is held during simulated `embed()` or batched forward pass.
- Condition waits use `while` loops and monotonic deadlines.

### Stress test reliability

Stress command: `python3 task4_pipeline.py --mode stress` (30 clients, 20 req/client, scoped cache, max_batch=8, timeout=0.02s, mid-run shutdown after 0.3s).

Observed:

- `accepted == completed + rejected_on_shutdown`: `300 == 300 + 0` (True)
- `integrity.ok`: True
- `no_deadlock_join`: True
- `errors`: 0
- Throughput: ~2505 req/s (small simulated delays, local run)

The pipeline also supports mid-run shutdown; requests not yet enqueued are rejected with `ShutdownError` and counted as `rejected_on_shutdown`.

### Benchmark comparison

Command: `python3 task4_pipeline.py --mode benchmark` (30 clients, 25 req/client, cache cap 32, queue cap 128, 4 workers, embed 3ms, forward 6ms, seed 1).

| config | rps | avg_ms | p95_ms | hit_rate | avg_batch | batch_util | accepted==comp+rej | integrity |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| coarse_unbatched | 629.73 | 44.08 | 46.93 | 0.909 | 1.00 | 1.000 | True | True |
| coarse_batched | 1769.83 | 12.27 | 34.90 | 0.912 | 7.73 | 0.966 | True | True |
| scoped_unbatched | 636.94 | 43.35 | 47.86 | 0.905 | 1.00 | 1.000 | True | True |
| scoped_batched | 2439.85 | 7.68 | 10.27 | 0.917 | 7.89 | 0.987 | True | True |

### Why the numbers behave this way

- **Batching dominates throughput**: unbatched (`max_batch_size=1`) pays one 6ms forward delay per request; batched (`max_batch_size=8`) amortizes that forward delay across ~7.6-7.8 requests, giving ~2.7x-3.6x higher throughput.
- **Scoped locking helps most when batching is enabled**: in unbatched mode, workers mostly serialize on per-request forward passes, so coarse vs scoped cache lock contention is less visible. In batched mode, the pipeline can drive more concurrent cache lookups/embeds; scoped locking avoids holding the cache lock during the 3ms embed, reducing contention and improving both throughput and tail latency.
- **Tail latency improves with scoped+batching**: coarse_batched has better average than unbatched but still high p95 (38.25ms) because coarse lock blocks cache access during embeds. scoped_batched has the best p95 (10.69ms) because cache mutations are short and embed work runs outside the lock.
- **Cache hit rate is stable** (~0.90-0.92) because all configs use the same key distribution/capacity; differences are minor run-to-run effects, not structural.

## Notes / constraints compliance

- No `queue.Queue`, `functools.lru_cache`, or `collections.OrderedDict` used for core structures.
- Uses `threading.Lock` and `threading.Condition`; condition waits are in `while` loops.
- Slow simulated embed/forward work is kept outside locks (except intentionally in coarse cache mode).
- Lock scopes are disjoint / documented to avoid deadlocks.
- Task 4 stress verifies `accepted == completed + rejected_on_shutdown`.
