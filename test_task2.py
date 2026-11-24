import random
import threading
import time

from task2_lru_cache import LRUEmbeddingCache, simulated_embed


def test_basic_lru():
    c = LRUEmbeddingCache(capacity=2, mode="scoped")
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # evicts b
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3
    ok, msg = c.verify_integrity()
    assert ok, msg


def stress_cache(mode: str):
    cap = 16
    cache = LRUEmbeddingCache(capacity=cap, mode=mode)
    keys = [f"k{i}" for i in range(40)]
    errors = []
    lock = threading.Lock()

    def worker(wid: int):
        rng = random.Random(100 + wid)
        for _ in range(300):
            k = rng.choice(keys)
            if rng.random() < 0.7:
                try:
                    val, _hit = cache.get_or_compute(k, lambda k=k: simulated_embed(k, 0.001))
                    if val != f"vec:{k}":
                        with lock:
                            errors.append(f"bad value {val} for {k}")
                except Exception as e:
                    with lock:
                        errors.append(repr(e))
            else:
                try:
                    cache.put(k, f"vec:{k}")
                except Exception as e:
                    with lock:
                        errors.append(repr(e))
            if rng.random() < 0.05:
                ok, msg = cache.verify_integrity()
                if not ok:
                    with lock:
                        errors.append(msg)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    ok, msg = cache.verify_integrity()
    assert ok, msg
    assert cache.size() <= cap
    stats = cache.snapshot_stats()
    return {"mode": mode, "ok": ok, "msg": msg, "errors": errors, "stats": stats}


if __name__ == "__main__":
    test_basic_lru()
    for mode in ["coarse", "scoped"]:
        r = stress_cache(mode)
        print(mode, r["stats"], "errors", len(r["errors"]))
        if r["errors"]:
            print(r["errors"][:5])
    print("task2 ok")
