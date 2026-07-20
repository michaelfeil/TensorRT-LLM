"""Regression tests for slot re-acquisition in prepare_resources.

A request resumed after an asynchronous KV-connector onboard (e.g. a
Dynamo KVBM host-cache load) re-enters scheduling with
``is_first_context_chunk=True`` even though its slot was already
allocated on the first pass. ``prepare_resources`` must reuse the
existing slot instead of calling ``add_slot`` again, which asserts on
duplicate request ids.
"""

import unittest
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager
from tensorrt_llm._torch.speculative.mtp import MTPHiddenStatesManager
from tensorrt_llm.llmapi import EagleDecodingConfig, MTPDecodingConfig


def _context_batch(request_id: int) -> ScheduledRequests:
    req = SimpleNamespace(request_id=request_id, is_first_context_chunk=True)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [req]
    return batch


class TestSpecSlotReuseOnReentry(unittest.TestCase):
    def _check_reentry(self, manager):
        manager.prepare_resources(_context_batch(request_id=7))
        slot_id = manager.slot_manager.get_slot(7)
        self.assertIsNotNone(slot_id)
        num_free = len(manager.slot_manager.free_slots)

        # The same request re-enters scheduling with
        # is_first_context_chunk=True after an async KV-connector onboard.
        manager.prepare_resources(_context_batch(request_id=7))
        self.assertEqual(manager.slot_manager.get_slot(7), slot_id)
        self.assertEqual(len(manager.slot_manager.free_slots), num_free)

    def test_eagle3_resource_manager(self):
        manager = Eagle3ResourceManager(
            EagleDecodingConfig(max_draft_len=3, speculative_model_dir="/does/not/matter"),
            dtype=torch.float16,
            hidden_size=16,
            max_num_requests=2,
            max_seq_len=8,
            max_num_tokens=32,
        )
        self._check_reentry(manager)

    def test_mtp_hidden_states_manager(self):
        manager = MTPHiddenStatesManager(
            MTPDecodingConfig(max_draft_len=1),
            dtype=torch.float16,
            hidden_size=16,
            max_num_requests=2,
        )
        self._check_reentry(manager)


if __name__ == "__main__":
    unittest.main()
