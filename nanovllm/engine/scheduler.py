from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.enable_chunked_prefill:
            return self._schedule_chunked()
        return self._schedule_default()

    def _schedule_default(self) -> tuple[list[Sequence], bool]:
        # prefill
        # Nano-VLLM优先调度prefill任务，然后是decode任务，且两者不混合
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 批次服务seq请求
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 判断加上seq是否超出能够serving的最大seq数 以及 是否有足够显存做kv cache
            # nanoVLLM的kvcache还搭配paged attention，所以要计算block数量
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            seq.num_computed_tokens = seq.num_cached_tokens
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        return self._schedule_decode()

    def _schedule_chunked(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # Phase 1: Continue ongoing prefills (running sequences still mid-prefill)
        for seq in list(self.running):
            if seq.is_prefill_finished:
                continue
            if num_seqs >= self.max_num_seqs:
                break
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            chunk = min(seq.num_uncomputed_tokens, remaining)
            num_batched_tokens += chunk
            num_seqs += 1
            scheduled_seqs.append(seq)

        # Phase 2: Admit new sequences from waiting
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            seq.num_computed_tokens = seq.num_cached_tokens
            chunk = min(seq.num_uncomputed_tokens, remaining)
            num_batched_tokens += chunk
            num_seqs += 1
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # Phase 3: Decode (only reached when no prefill work is pending)
        return self._schedule_decode()

    def _schedule_decode(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 抢占逻辑，如果显存空间不够，则抢占别的序列，直到空间足够
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 将KV Cache换出到cpu或直接丢弃
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def postprocess_chunked_prefill(self, seqs: list[Sequence], token_ids: list[int]):
        """Post-process after a chunked prefill step.
        token_ids[i] is None for mid-prefill sequences, valid int for completing ones.
        """
        for seq, token_id in zip(seqs, token_ids):
            if token_id is None:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
