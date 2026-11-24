import threading
import time

from task3_batch_queue import DynamicBatchQueue, ShutdownError


def test_basic():
    q = DynamicBatchQueue[int](capacity=4)
    q.put(1)
    q.put(2)
    b = q.get_batch(10, 0.05)
    assert b == [1, 2]


def stress_queue():
    q = DynamicBatchQueue[int](capacity=16)
    produced = 0
    consumed = 0
    lock = threading.Lock()
    errors = []
    stop = threading.Event()
    total_items = 2000

    def producer(pid):
        nonlocal produced
        for i in range(total_items // 10):
            item = pid * 100000 + i
            try:
                q.put(item)
                with lock:
                    produced += 1
            except ShutdownError:
                return
            except Exception as e:
                with lock:
                    errors.append(repr(e))

    def consumer():
        nonlocal consumed
        while True:
            batch = q.get_batch(8, 0.02)
            if not batch:
                if q.is_shutdown() and q.qsize() == 0:
                    return
                continue
            with lock:
                consumed += len(batch)

    producers = [threading.Thread(target=producer, args=(i,)) for i in range(10)]
    consumers = [threading.Thread(target=consumer) for _ in range(4)]
    for t in consumers: t.start()
    for t in producers: t.start()
    for t in producers: t.join()
    # allow drain
    time.sleep(0.2)
    q.shutdown()
    for t in consumers: t.join(timeout=5)
    with lock:
        return {"produced": produced, "consumed": consumed, "errors": errors, "stats": q.stats(), "consumer_alive": [t.is_alive() for t in consumers]}


if __name__ == "__main__":
    test_basic()
    r = stress_queue()
    print(r)
    assert not any(r["consumer_alive"])
    assert r["produced"] == r["consumed"]
    assert not r["errors"]
    print("task3 ok")
