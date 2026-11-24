"""Task 2: Thread-safe LRU Embedding Cache built from scratch.

No queue.Queue, functools.lru_cache, or collections.OrderedDict used.
Uses threading.Lock/Condition.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _Node:
    key: Any
    value: Any
    prev: "_Node | None" = None
    next: "_Node | None" = None


class _InFlight:
    __slots__ = ("event", "done", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.done = False
        self.result: Any = None
        self.error: Exception | None = None


class LRUEmbeddingCache:
    """Doubly-linked-list + hashmap LRU cache with coarse/scoped lock modes."""

    def __init__(self, capacity: int, mode: str = "scoped"):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        if mode not in {"coarse", "scoped"}:
            raise ValueError("mode must be 'coarse' or 'scoped'")
        self.capacity = capacity
        self.mode = mode
        self._map: dict[Any, _Node] = {}
        self._head = _Node("__head__", None)  # MRU side sentinel
        self._tail = _Node("__tail__", None)  # LRU side sentinel
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ---- internal linked list helpers (caller must hold lock) ----
    def _remove_node(self, node: _Node) -> None:
        prev_node = node.prev
        next_node = node.next
        assert prev_node is not None and next_node is not None
        prev_node.next = next_node
        next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _add_to_front(self, node: _Node) -> None:
        first = self._head.next
        assert first is not None
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node

    def _move_to_front(self, node: _Node) -> None:
        self._remove_node(node)
        self._add_to_front(node)

    def _evict_lru(self) -> _Node | None:
        lru = self._tail.prev
        if lru is None or lru is self._head:
            return None
        self._remove_node(lru)
        self._map.pop(lru.key, None)
        self._size -= 1
        self.evictions += 1
        return lru

    def _put_locked(self, key: Any, value: Any) -> None:
        if self.capacity == 0:
            return
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._move_to_front(node)
            return
        new_node = _Node(key, value)
        self._add_to_front(new_node)
        self._map[key] = new_node
        self._size += 1
        while self._size > self.capacity:
            self._evict_lru()

    # ---- public API ----
    def get(self, key: Any) -> Any | None:
        with self._lock:
            node = self._map.get(key)
            if node is None:
                self.misses += 1
                return None
            if isinstance(node.value, _InFlight):
                self.misses += 1
                return None
            self._move_to_front(node)
            self.hits += 1
            return node.value

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            # Do not overwrite in-flight placeholder blindly; waiters rely on it.
            node = self._map.get(key)
            if node is not None and isinstance(node.value, _InFlight):
                inflight = node.value
                node.value = value
                self._move_to_front(node)
                inflight.result = value
                inflight.done = True
                inflight.event.set()
                return
            self._put_locked(key, value)

    def get_or_compute(self, key: Any, compute_fn: Callable[[], Any]) -> tuple[Any, bool]:
        """Return (value, was_hit)."""
        if self.mode == "coarse":
            return self._get_or_compute_coarse(key, compute_fn)
        return self._get_or_compute_scoped(key, compute_fn)

    def _get_or_compute_coarse(self, key: Any, compute_fn: Callable[[], Any]) -> tuple[Any, bool]:
        with self._lock:
            node = self._map.get(key)
            if node is not None and not isinstance(node.value, _InFlight):
                self._move_to_front(node)
                self.hits += 1
                return node.value, True
            self.misses += 1
            value = compute_fn()  # intentionally inside lock for coarse baseline
            self._put_locked(key, value)
            return value, False

    def _get_or_compute_scoped(self, key: Any, compute_fn: Callable[[], Any]) -> tuple[Any, bool]:
        while True:
            with self._lock:
                node = self._map.get(key)
                if node is not None and not isinstance(node.value, _InFlight):
                    self._move_to_front(node)
                    self.hits += 1
                    return node.value, True

                if node is not None and isinstance(node.value, _InFlight):
                    inflight = node.value
                    waiter = True
                else:
                    # Become producer: insert in-flight placeholder.
                    if self.capacity == 0:
                        self.misses += 1
                        # No storage possible; compute outside lock, no caching.
                        producer_no_cache = True
                        inflight = None
                        waiter = False
                    else:
                        inflight = _InFlight()
                        placeholder = _Node(key, inflight)
                        self._add_to_front(placeholder)
                        self._map[key] = placeholder
                        self._size += 1
                        while self._size > self.capacity:
                            self._evict_lru()
                        self.misses += 1
                        producer_no_cache = False
                        waiter = False

            if waiter:
                assert inflight is not None
                inflight.event.wait()
                # Loop to re-check cache state.
                if inflight.error is not None:
                    # Producer failed; retry as potential producer.
                    continue
                # Value may have been evicted after completion; re-check.
                continue

            # Producer path (outside lock)
            if producer_no_cache:
                value = compute_fn()
                return value, False

            assert inflight is not None
            try:
                value = compute_fn()
            except Exception as e:
                with self._lock:
                    cur = self._map.get(key)
                    if cur is not None and cur.value is inflight:
                        self._remove_node(cur)
                        self._map.pop(key, None)
                        self._size -= 1
                    inflight.error = e
                    inflight.done = True
                    inflight.event.set()
                raise

            with self._lock:
                cur = self._map.get(key)
                if cur is not None and cur.value is inflight:
                    cur.value = value
                    self._move_to_front(cur)
                elif self.capacity > 0:
                    # Placeholder evicted/replaced; cache computed value anyway.
                    self._put_locked(key, value)
                inflight.result = value
                inflight.done = True
                inflight.event.set()
            return value, False

    def size(self) -> int:
        with self._lock:
            return self._size

    def snapshot_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "size": self._size,
                "capacity": self.capacity,
            }

    def verify_integrity(self) -> tuple[bool, str]:
        with self._lock:
            return self._verify_integrity_locked()

    def _verify_integrity_locked(self) -> tuple[bool, str]:
        if self._size < 0 or self._size > self.capacity:
            return False, f"size out of range: size={self._size} capacity={self.capacity}"
        if len(self._map) != self._size:
            return False, f"map size {len(self._map)} != tracked size {self._size}"

        count = 0
        seen: set[Any] = set()
        node = self._head.next
        prev = self._head
        while node is not None and node is not self._tail:
            if node.prev is not prev:
                return False, f"broken prev pointer at key={node.key!r}"
            if node.key in seen:
                return False, f"duplicate key in list: {node.key!r}"
            seen.add(node.key)
            map_node = self._map.get(node.key)
            if map_node is not node:
                return False, f"map/list mismatch at key={node.key!r}"
            count += 1
            if count > self._size + 5:
                return False, "cycle or runaway traversal detected"
            prev = node
            node = node.next
        if node is not self._tail:
            return False, "forward traversal did not reach tail"
        if self._tail.prev is not prev:
            return False, "tail.prev mismatch after forward traversal"
        if count != self._size:
            return False, f"forward count {count} != size {self._size}"

        count_b = 0
        node = self._tail.prev
        nxt = self._tail
        while node is not None and node is not self._head:
            if node.next is not nxt:
                return False, f"broken next pointer at key={node.key!r}"
            count_b += 1
            if count_b > self._size + 5:
                return False, "cycle in backward traversal"
            nxt = node
            node = node.prev
        if node is not self._head:
            return False, "backward traversal did not reach head"
        if count_b != self._size:
            return False, f"backward count {count_b} != size {self._size}"
        return True, "ok"


def simulated_embed(key: Any, delay_s: float = 0.001) -> str:
    time.sleep(delay_s)
    return f"vec:{key}"
