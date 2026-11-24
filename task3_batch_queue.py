"""Task 3: Bounded Dynamic Inference Batching Queue.

Built from scratch with threading.Lock and threading.Condition.
No queue.Queue, functools.lru_cache, or collections.OrderedDict.
"""
from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class ShutdownError(RuntimeError):
    pass


class DynamicBatchQueue(Generic[T]):
    """Bounded buffer with blocking put and dynamic batch get."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self._buf: list[T] = []
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._shutdown = False
        # metrics
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.rejected_on_shutdown = 0

    def put(self, item: T) -> None:
        """Block until space is available, then enqueue and notify consumers."""
        with self._cond:
            while len(self._buf) >= self.capacity and not self._shutdown:
                self._cond.wait()
            if self._shutdown:
                self.rejected_on_shutdown += 1
                raise ShutdownError("queue is shut down")
            self._buf.append(item)
            self.total_enqueued += 1
            self._cond.notify_all()

    def offer(self, item: T, timeout_s: float | None = None) -> bool:
        """Optional helper: try to enqueue within timeout. Not required but useful."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cond:
            while len(self._buf) >= self.capacity and not self._shutdown:
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=remaining)
            if self._shutdown:
                self.rejected_on_shutdown += 1
                return False
            self._buf.append(item)
            self.total_enqueued += 1
            self._cond.notify_all()
            return True

    def get_batch(self, max_batch_size: int, timeout_s: float) -> list[T]:
        """Wait until batch full, timeout with >=1 item, or shutdown drain completes.

        Returns empty list only when shutdown and queue is drained.
        """
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while len(self._buf) == 0 and not self._shutdown:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

            # If timeout expired but we have items, return them. If batch full, return.
            # Also handle case where we woke up and need to check batch/timeout conditions.
            while len(self._buf) < max_batch_size and len(self._buf) > 0 and not self._shutdown:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

            # After shutdown, drain remaining items.
            if len(self._buf) == 0:
                return []

            n = min(max_batch_size, len(self._buf))
            batch = self._buf[:n]
            del self._buf[:n]
            self.total_dequeued += len(batch)
            self._cond.notify_all()  # wake producers waiting for space
            return batch

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()

    def is_shutdown(self) -> bool:
        with self._lock:
            return self._shutdown

    def qsize(self) -> int:
        with self._lock:
            return len(self._buf)

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "size": len(self._buf),
                "capacity": self.capacity,
                "shutdown": self._shutdown,
                "total_enqueued": self.total_enqueued,
                "total_dequeued": self.total_dequeued,
                "rejected_on_shutdown": self.rejected_on_shutdown,
            }
