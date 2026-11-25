# Concurrent ML Inference Server - Training Task Report

## Workspace artifacts

- `task1_baseline.py` - intentionally broken baseline router + reproducible stress script.
- `task2_lru_cache.py` - thread-safe LRU embedding cache built from scratch (coarse/scoped modes).
- `task3_batch_queue.py` - bounded dynamic batching queue with `Lock`/`Condition`.
- `task4_pipeline.py` - integrated pipeline, stress test, and benchmark comparisons; loads saved tinygrad model.
- `ml_model.py` - tinygrad model/tokenizer, save/load, embedding + batched inference helpers.
- `train_model.py` - trains and saves the tinygrad model to `artifacts/` (separate from benchmarks).
- `artifacts/tiny_embed_classifier.npz`, `artifacts/tiny_embed_classifier.json`, `artifacts/train_metrics.json` - saved model/metadata.
- `test_task2.py`, `test_task3.py` - correctness/stress checks for Task 2 and Task 3.
- `benchmark_results.json` - raw benchmark output.

## How to run

```bash
DEV=PYTHON .venv/bin/python train_model.py        # Train & save tinygrad model (separate script)
python3 task1_baseline.py                         # Task 1 reproducible stress + bug summary
python3 test_task2.py                             # Task 2 LRU correctness/stress
python3 test_task3.py                             # Task 3 queue correctness/stress
DEV=PYTHON .venv/bin/python task4_pipeline.py --mode stress     # Task 4 stress; loads saved model
DEV=PYTHON .venv/bin/python task4_pipeline.py --mode benchmark  # Task 4 benchmark; loads saved model
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

## ML model (tinygrad)

- **Training script**: `train_model.py` (separate). It builds a tiny synthetic prompt-classification dataset, trains a small tinygrad model, and saves it to `artifacts/`.
- **Model**: `TinyEmbedClassifier` in `ml_model.py` — token IDs (`nn.Embedding`) -> mean pool -> linear logits (`nn.Linear`). This is intentionally simple and gives a real **embedding stage** that benefits from caching.
- **Saved artifacts**: `artifacts/tiny_embed_classifier.npz`, `tiny_embed_classifier.json`, `train_metrics.json`. Benchmarks/stress tests load these via `load_model()` and **do not retrain**.
- **Inference path**: `embed_prompt()` computes prompt embedding with tinygrad; `Router` caches embeddings in the Task 2 LRU cache (`get_or_compute`); workers run batched `logits_from_embedding(Tensor(batch_embs))` for real forward passes.
- **Runtime note**: tinygrad runs with `DEV=PYTHON` in this environment for portability. A `model_lock` serializes tinygrad calls because the runtime/model state is shared; this lock is never held with cache/queue locks.

## Task 4: Integrated pipeline, stress test, and benchmarks

Implementation: `task4_pipeline.py`.

### Pipeline

`Client Threads -> Router -> Thread-Safe LRU Cache (embeddings) -> DynamicBatchQueue -> Inference Worker Pool`

- Router uses `cache.get_or_compute(prompt, embed_fn)`; on miss it computes a real tinygrad embedding. Scoped mode releases cache lock during embedding compute (stampede protected by in-flight placeholder).
- Accepted requests carry cached embedding to `DynamicBatchQueue`; workers pull batches and run a real batched tinygrad forward pass (`logits_from_embedding`) outside cache/queue locks.
- Request completion is signaled by per-request `threading.Event`.

### Lock hierarchy / deadlock prevention

1. **Router stats lock** (`Router._stats_lock`) - short-lived counters/latency list only.
2. **Cache lock** (inside `LRUEmbeddingCache`) - only fast map/list/in-flight mutations; never held during model inference.
3. **Queue condition lock** (inside `DynamicBatchQueue`) - guards bounded buffer and condition waits.
4. **Model inference lock** (`pipeline._model_lock`) - serializes tinygrad embed/forward calls; never held together with cache or queue locks.

Rules enforced:
- Cache and queue locks are disjoint; router never holds them simultaneously.
- No cache/queue lock is held during tinygrad embedding or batched forward pass.
- Condition waits use `while` loops and monotonic deadlines.

### Stress test reliability

Command: `DEV=PYTHON .venv/bin/python task4_pipeline.py --mode stress` (loads saved model; 30 clients × 10 req, scoped cache, max_batch=8, shutdown_after_s=0.5).

Observed (real tinygrad inference, saved model):

- `accepted == completed + rejected_on_shutdown`: `300 == 9 + 291` (`True`) — mid-run shutdown correctly rejects un-enqueued requests.
- `integrity.ok`: `True`
- `no_deadlock_join`: `True`
- `errors`: `0`; queue drained cleanly (`enqueued==dequeued==9`).

The high rejection count is due slow `DEV=PYTHON` tinygrad inference relative to the 0.5s shutdown window; it confirms shutdown accounting and no lost requests.

### Benchmark comparison

Command: `DEV=PYTHON .venv/bin/python task4_pipeline.py --mode benchmark` (20 clients × 10 req, cache cap 32, queue cap 128, 2 workers, saved model loaded once).

| config | rps | avg_ms | p95_ms | hit_rate | avg_batch | batch_util | accepted==comp+rej | integrity |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| coarse_unbatched | 19.70 | 973.22 | 1956.38 | 0.810 | 1.00 | 1.000 | True | True |
| coarse_batched | 20.93 | 907.85 | 1862.00 | 0.815 | 3.23 | 0.403 | True | True |
| scoped_unbatched | 20.66 | 927.28 | 2290.29 | 0.815 | 1.00 | 1.000 | True | True |
| scoped_batched | 22.41 | 833.24 | 1850.20 | 0.820 | 6.67 | 0.833 | True | True |

### Why the numbers behave this way

- **Real inference is the bottleneck**: with `DEV=PYTHON`, tinygrad embedding/forward calls are much slower than the old `time.sleep()` simulation, so throughput is lower and latency higher. This is expected for a portability backend.
- **Batching still helps**: `scoped_batched` has the best throughput and lowest average/p95 because batched forward passes amortize per-request overhead; average batch size/utilization rises to 6.67/0.833.
- **Scoped vs coarse**: scoped mode releases cache lock during embedding compute. The gain is modest here because the global `model_lock` serializes tinygrad execution, but scoped still avoids holding cache lock across inference and gives the best batched result.
- **Cache hit rate** remains ~0.81-0.82 across configs with same key distribution/capacity, confirming the LRU cache is reused effectively.

## Notes / constraints compliance

- No `queue.Queue`, `functools.lru_cache`, or `collections.OrderedDict` used for core structures.
- Uses `threading.Lock` and `threading.Condition`; condition waits are in `while` loops.
- Slow simulated embed/forward work is kept outside locks (except intentionally in coarse cache mode).
- Lock scopes are disjoint / documented to avoid deadlocks.
- Task 4 stress verifies `accepted == completed + rejected_on_shutdown`.
