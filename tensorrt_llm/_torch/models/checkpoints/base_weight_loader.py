# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import bisect
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tensorrt_llm.mapping import Mapping


class ConsumableWeightsDict:
    """
    Wrapper around a weights dictionary that allows marking keys as consumed
    to free memory during model loading.

    This reduces peak memory usage by deleting weight tensors from the dictionary
    after they have been copied to the model, rather than keeping all weights
    in memory until loading completes.

    Thread-safe: uses a lock to protect concurrent access. Iteration methods
    (keys, values, items, __iter__) return snapshot copies to allow safe
    concurrent iteration while other threads may modify the dictionary.

    Prefix lookups (prefix_items, mark_consumed) go through a lazily built
    sorted-key index: per-module weight filtering during load is O(log n + m)
    instead of a full O(n) key scan, which dominates load time for large MoE
    checkpoints (~1e5 keys x ~2e3 modules).
    """

    def __init__(self, weights: Dict[str, Any]):
        self._weights = weights
        self._lock = threading.Lock()
        self._sorted_keys: Optional[List[str]] = None

    def _sorted_keys_locked(self) -> List[str]:
        if self._sorted_keys is None:
            self._sorted_keys = sorted(self._weights)
        return self._sorted_keys

    @staticmethod
    def _prefix_end(prefix: str) -> str:
        # Smallest string ordered after every string with this prefix.
        return prefix[:-1] + chr(ord(prefix[-1]) +
                                 1) if prefix else chr(0x10FFFF)

    def prefix_items(self, prefix: str) -> List[Tuple[str, Any]]:
        """(key, value) pairs for keys starting with prefix.

        Same result set as filtering items() with str.startswith, but via
        bisect on the sorted-key index. Deleted keys may linger in the index,
        so membership is re-checked against the live dict.
        """
        with self._lock:
            sorted_keys = self._sorted_keys_locked()
            lo = bisect.bisect_left(sorted_keys, prefix)
            hi = bisect.bisect_left(sorted_keys, self._prefix_end(prefix), lo)
            weights = self._weights
            return [(k, weights[k]) for k in sorted_keys[lo:hi] if k in weights]

    def __getitem__(self, key: str) -> Any:
        return self._weights[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if self._sorted_keys is not None and key not in self._weights:
                bisect.insort(self._sorted_keys, key)
            self._weights[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._weights[key]
            self._sorted_keys = None

    def __contains__(self, key: str) -> bool:
        return key in self._weights

    def __len__(self) -> int:
        return len(self._weights)

    def __iter__(self) -> Iterator[str]:
        # Return iterator over a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return iter(list(self._weights.keys()))

    def keys(self):
        # Return a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return list(self._weights.keys())

    def values(self):
        # Return a snapshot copy of values to allow concurrent modification
        with self._lock:
            return list(self._weights.values())

    def items(self) -> Iterator[Tuple[str, Any]]:
        # Return a snapshot copy of items to allow concurrent modification
        with self._lock:
            return list(self._weights.items())

    def get(self, key: str, default: Any = None) -> Any:
        return self._weights.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            self._weights.update(other)
            # Bulk load path; rebuild the index lazily on next prefix lookup.
            self._sorted_keys = None

    def mark_consumed(self, prefix: str) -> int:
        """
        Delete all keys starting with the given prefix to free memory.

        Args:
            prefix: The prefix to match. Keys starting with "{prefix}." will be deleted.

        Returns:
            The number of keys deleted.

        Thread-safe: uses a lock to prevent concurrent modification issues.
        """
        with self._lock:
            sorted_keys = self._sorted_keys_locked()
            dotted = prefix + "."
            lo = bisect.bisect_left(sorted_keys, dotted)
            hi = bisect.bisect_left(sorted_keys, self._prefix_end(dotted), lo)
            deleted = 0
            for key in sorted_keys[lo:hi]:
                if key in self._weights:
                    del self._weights[key]
                    deleted += 1
            del sorted_keys[lo:hi]
            return deleted


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> Union[Dict[str, Any], ConsumableWeightsDict]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.
            mapping: A mapping object containing the distributed configuration.
            **kwargs: Optional format-specific loader arguments.

        Returns:
            A dictionary (or ConsumableWeightsDict) where keys are tensor names
            and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
