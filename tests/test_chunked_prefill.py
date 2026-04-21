"""Unit tests for chunked prefill.

Tests the scheduling, sequence state, and block management logic
without requiring a GPU or model weights.
"""
import pytest
from collections import deque
from copy import deepcopy

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


# ---------------------------------------------------------------------------
# Sequence property tests
# ---------------------------------------------------------------------------

class TestSequenceChunkedPrefill:

    def test_initial_state(self):
        seq = Sequence([1, 2, 3, 4, 5])
        assert seq.num_computed_tokens == 0
        assert seq.num_uncomputed_tokens == 5
        assert not seq.is_prefill_finished

    def test_after_partial_compute(self):
        seq = Sequence([1, 2, 3, 4, 5])
        seq.num_computed_tokens = 3
        assert seq.num_uncomputed_tokens == 2
        assert not seq.is_prefill_finished

    def test_after_full_compute(self):
        seq = Sequence([1, 2, 3, 4, 5])
        seq.num_computed_tokens = 5
        assert seq.num_uncomputed_tokens == 0
        assert seq.is_prefill_finished

    def test_with_cached_tokens(self):
        seq = Sequence([1, 2, 3, 4, 5])
        seq.num_cached_tokens = 2
        seq.num_computed_tokens = 2
        assert seq.num_uncomputed_tokens == 3
        assert not seq.is_prefill_finished

    def test_pickle_roundtrip(self):
        """Ensure num_computed_tokens survives pickle for TP serialization."""
        import pickle
        seq = Sequence([10, 20, 30, 40])
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 2
        seq.block_table = [0, 1]
        data = pickle.dumps(seq)
        restored = pickle.loads(data)
        assert restored.num_computed_tokens == 2
        assert restored.num_prompt_tokens == 4


# ---------------------------------------------------------------------------
# BlockManager deallocate tests
# ---------------------------------------------------------------------------

class TestBlockManagerDeallocate:

    def test_deallocate_resets_computed_tokens(self):
        bm = BlockManager(num_blocks=16, block_size=256)
        seq = Sequence(list(range(300)))
        bm.allocate(seq)
        seq.num_computed_tokens = seq.num_cached_tokens
        seq.num_computed_tokens += 256
        assert seq.num_computed_tokens > 0
        bm.deallocate(seq)
        assert seq.num_computed_tokens == 0
        assert seq.num_cached_tokens == 0
        assert seq.block_table == []


# ---------------------------------------------------------------------------
# Scheduler tests (mock block manager for simplicity)
# ---------------------------------------------------------------------------

class MockConfig:
    """Minimal config stub for Scheduler."""
    def __init__(self, *, max_num_seqs=4, max_num_batched_tokens=1024,
                 num_kvcache_blocks=64, kvcache_block_size=256,
                 enable_chunked_prefill=False, eos=-1):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_kvcache_blocks = num_kvcache_blocks
        self.kvcache_block_size = kvcache_block_size
        self.enable_chunked_prefill = enable_chunked_prefill
        self.eos = eos


class TestSchedulerDefault:
    """Verify the non-chunked path still works after refactoring."""

    def test_prefill_then_decode(self):
        from nanovllm.engine.scheduler import Scheduler
        config = MockConfig(max_num_batched_tokens=4096, num_kvcache_blocks=64)
        sched = Scheduler(config)
        seq = Sequence(list(range(512)))
        sched.add(seq)

        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert len(seqs) == 1
        assert seqs[0].status == SequenceStatus.RUNNING
        assert seqs[0].num_computed_tokens == seqs[0].num_cached_tokens

    def test_multiple_prefill(self):
        from nanovllm.engine.scheduler import Scheduler
        config = MockConfig(max_num_batched_tokens=4096, num_kvcache_blocks=64)
        sched = Scheduler(config)
        for _ in range(3):
            sched.add(Sequence(list(range(256))))

        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert len(seqs) == 3


class TestSchedulerChunkedPrefill:

    def _make_scheduler(self, max_num_batched_tokens=512, num_kvcache_blocks=64):
        config = MockConfig(
            max_num_batched_tokens=max_num_batched_tokens,
            num_kvcache_blocks=num_kvcache_blocks,
            enable_chunked_prefill=True,
        )
        from nanovllm.engine.scheduler import Scheduler
        return Scheduler(config)

    def test_short_prompt_single_chunk(self):
        """Prompt shorter than budget processes in one chunk."""
        sched = self._make_scheduler(max_num_batched_tokens=1024)
        seq = Sequence(list(range(300)))
        sched.add(seq)

        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert len(seqs) == 1
        assert seqs[0].num_computed_tokens == seqs[0].num_cached_tokens

    def test_long_prompt_multi_chunk(self):
        """Prompt longer than budget requires multiple schedule() calls."""
        sched = self._make_scheduler(max_num_batched_tokens=512)
        seq = Sequence(list(range(1500)))
        sched.add(seq)

        seqs1, is_prefill1 = sched.schedule()
        assert is_prefill1
        assert len(seqs1) == 1
        assert seqs1[0].status == SequenceStatus.RUNNING
        # Simulate model_runner updating num_computed_tokens for first chunk
        seqs1[0].num_computed_tokens = seqs1[0].num_cached_tokens + 512
        assert not seqs1[0].is_prefill_finished

        seqs2, is_prefill2 = sched.schedule()
        assert is_prefill2
        assert len(seqs2) == 1
        assert seqs2[0] is seqs1[0]

    def test_chunked_falls_through_to_decode(self):
        """After prefill completes, schedule returns decode."""
        sched = self._make_scheduler(max_num_batched_tokens=2048)
        seq = Sequence(list(range(300)))
        sched.add(seq)

        seqs, is_prefill = sched.schedule()
        assert is_prefill
        # Simulate full prefill completion
        seqs[0].num_computed_tokens = seqs[0].num_prompt_tokens
        # Simulate postprocess: append one token
        seqs[0].append_token(99)

        seqs2, is_prefill2 = sched.schedule()
        assert not is_prefill2
        assert len(seqs2) == 1

    def test_budget_split_across_sequences(self):
        """Multiple sequences share the token budget."""
        sched = self._make_scheduler(max_num_batched_tokens=768)
        seq1 = Sequence(list(range(500)))
        seq2 = Sequence(list(range(500)))
        sched.add(seq1)
        sched.add(seq2)

        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert len(seqs) == 2
        # seq1 gets min(500, 768) = 500 tokens
        # seq2 gets min(500, 768-500) = 268 tokens
        # Both are scheduled, seq2 won't finish prefill in one step

    def test_ongoing_prefill_has_priority(self):
        """Ongoing prefills in running queue are scheduled before new waits."""
        sched = self._make_scheduler(max_num_batched_tokens=512)
        seq1 = Sequence(list(range(1000)))
        sched.add(seq1)

        # First schedule: admit seq1, process first chunk
        seqs, _ = sched.schedule()
        seqs[0].num_computed_tokens = seqs[0].num_cached_tokens + 512

        # Add a new sequence while seq1 is mid-prefill
        seq2 = Sequence(list(range(200)))
        sched.add(seq2)

        # Second schedule: should continue seq1 first
        seqs2, is_prefill = sched.schedule()
        assert is_prefill
        assert seq1 in seqs2

    def test_preempted_sequence_resets_computed(self):
        """Preemption resets num_computed_tokens so recompute starts fresh."""
        sched = self._make_scheduler(max_num_batched_tokens=1024, num_kvcache_blocks=4)
        seq = Sequence(list(range(300)))
        sched.add(seq)

        seqs, _ = sched.schedule()
        seqs[0].num_computed_tokens = 256

        sched.preempt(seqs[0])
        assert seqs[0].num_computed_tokens == 0
        assert seqs[0].num_cached_tokens == 0
        assert seqs[0].status == SequenceStatus.WAITING


# ---------------------------------------------------------------------------
# Slot mapping correctness tests
# ---------------------------------------------------------------------------

class TestSlotMappingChunked:
    """Verify slot_mapping is correct for chunked prefill.

    These tests use a mock that simulates prepare_prefill's slot_mapping
    logic without CUDA tensors.
    """

    @staticmethod
    def compute_slot_mapping(block_table, block_size, start, end):
        """Replicate prepare_prefill's slot_mapping computation."""
        if start >= end:
            return []
        slots = []
        start_block_idx = start // block_size
        end_block_idx = (end - 1) // block_size
        for i in range(start_block_idx, end_block_idx + 1):
            block_id = block_table[i]
            block_token_start = i * block_size
            in_block_start = max(0, start - block_token_start)
            in_block_end = min(block_size, end - block_token_start)
            slot_start = block_id * block_size + in_block_start
            slot_end = block_id * block_size + in_block_end
            slots.extend(range(slot_start, slot_end))
        return slots

    def test_full_prefill_matches_original(self):
        """Full prefill starting at 0 should match the original code's behavior."""
        block_table = [3, 7, 1]
        block_size = 4
        slots = self.compute_slot_mapping(block_table, block_size, 0, 10)
        # Block 3: slots 12,13,14,15 (4 tokens)
        # Block 7: slots 28,29,30,31 (4 tokens)
        # Block 1: slots 4,5         (2 tokens, partial)
        expected = list(range(12, 16)) + list(range(28, 32)) + list(range(4, 6))
        assert slots == expected

    def test_second_chunk_block_aligned(self):
        """Second chunk starting at a block boundary."""
        block_table = [3, 7, 1]
        block_size = 4
        slots = self.compute_slot_mapping(block_table, block_size, 4, 10)
        # Block 7: slots 28,29,30,31 (4 tokens)
        # Block 1: slots 4,5         (2 tokens, partial)
        expected = list(range(28, 32)) + list(range(4, 6))
        assert slots == expected

    def test_chunk_mid_block(self):
        """Chunk starting in the middle of a block."""
        block_table = [3, 7, 1]
        block_size = 4
        slots = self.compute_slot_mapping(block_table, block_size, 2, 6)
        # Block 3: positions 2,3 → slots 14,15
        # Block 7: positions 0,1 → slots 28,29
        expected = [14, 15, 28, 29]
        assert slots == expected

    def test_single_token_chunk(self):
        """Chunk of exactly 1 token."""
        block_table = [5, 2]
        block_size = 4
        slots = self.compute_slot_mapping(block_table, block_size, 3, 4)
        # Block 5, position 3 → slot 23
        assert slots == [23]

    def test_chunk_spanning_three_blocks(self):
        """Chunk that spans from mid-block through a full block to mid-block."""
        block_table = [0, 1, 2, 3]
        block_size = 4
        slots = self.compute_slot_mapping(block_table, block_size, 2, 14)
        # Block 0: positions 2,3 → slots 2,3
        # Block 1: positions 0-3 → slots 4,5,6,7
        # Block 2: positions 0-3 → slots 8,9,10,11
        # Block 3: positions 0,1 → slots 12,13
        expected = [2, 3] + list(range(4, 8)) + list(range(8, 12)) + [12, 13]
        assert slots == expected

    def test_two_chunks_cover_all_slots(self):
        """Two consecutive chunks should cover exactly the same slots as one full prefill."""
        block_table = [5, 2, 9]
        block_size = 4
        full_slots = self.compute_slot_mapping(block_table, block_size, 0, 10)

        chunk1_slots = self.compute_slot_mapping(block_table, block_size, 0, 6)
        chunk2_slots = self.compute_slot_mapping(block_table, block_size, 6, 10)
        assert chunk1_slots + chunk2_slots == full_slots

    def test_three_chunks_cover_all_slots(self):
        """Three consecutive chunks with non-aligned boundaries."""
        block_table = [10, 20, 30, 40]
        block_size = 4
        full_slots = self.compute_slot_mapping(block_table, block_size, 0, 15)

        c1 = self.compute_slot_mapping(block_table, block_size, 0, 5)
        c2 = self.compute_slot_mapping(block_table, block_size, 5, 11)
        c3 = self.compute_slot_mapping(block_table, block_size, 11, 15)
        assert c1 + c2 + c3 == full_slots


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
