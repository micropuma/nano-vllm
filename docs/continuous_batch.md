# NanoVLLM Continuous Batching 深度解析

> 基于 NanoVLLM 源码逐行分析，结合 FlashAttention / Paged KV Cache 原理，完整还原 Continuous Batching 的调度与计算过程。 完全由 Sonnet4.6生成

---

## 目录

1. [什么是 Continuous Batching](#1-什么是-continuous-batching)
2. [整体架构概览](#2-整体架构概览)
3. [核心数据结构](#3-核心数据结构)
   - 3.1 [Sequence](#31-sequence)
   - 3.2 [Block 与 BlockManager](#32-block-与-blockmanager)
   - 3.3 [Context 全局上下文](#33-context-全局上下文)
4. [调度器：Scheduler 全流程](#4-调度器scheduler-全流程)
   - 4.1 [核心队列设计](#41-核心队列设计)
   - 4.2 [Prefill 调度](#42-prefill-调度)
   - 4.3 [Decode 调度与抢占](#43-decode-调度与抢占)
   - 4.4 [后处理 postprocess](#44-后处理-postprocess)
5. [ModelRunner：Batch 组装](#5-modelrunnerbatch-组装)
   - 5.1 [prepare\_prefill](#51-prepare_prefill)
   - 5.2 [prepare\_decode](#52-prepare_decode)
6. [Batch 如何喂给模型](#6-batch-如何喂给模型)
   - 6.1 [Embedding 层](#61-embedding-层)
   - 6.2 [MLP 层 —— 无需感知 Batch 边界](#62-mlp-层--无需感知-batch-边界)
   - 6.3 [Attention 层 —— Prefill 阶段](#63-attention-层--prefill-阶段)
   - 6.4 [Attention 层 —— Decode 阶段](#64-attention-层--decode-阶段)
7. [KV Cache 写入：store\_kvcache Triton Kernel](#7-kv-cache-写入store_kvcache-triton-kernel)
8. [Paged Attention + Prefix Cache](#8-paged-attention--prefix-cache)
9. [CUDAGraph 加速 Decode](#9-cudagraph-加速-decode)
10. [Tensor Parallelism 中的 Batch 拼接](#10-tensor-parallelism-中的-batch-拼接)
11. [端到端数据流图](#11-端到端数据流图)

---

## 1. 什么是 Continuous Batching

传统静态批处理（Static Batching）要求一个 batch 内所有序列同时完成，批次间存在大量等待（最短序列等最长序列），GPU 利用率低下。

**Continuous Batching**（也叫 Iteration-level scheduling）的核心思想：

- 每次迭代（每生成一个 token）之后重新调度，已完成的序列立刻离队，新请求随时插入。
- 将 **Prefill**（处理 prompt）和 **Decode**（逐步生成）分开调度，每次迭代只做其中一种。
- 通过 **Paged KV Cache** 让不连续的物理内存可以服务不同长度的序列，彻底消除碎片。

NanoVLLM 实现了上述机制，调度粒度是每一次 `step()`，由 [`nanovllm/engine/llm_engine.py`](../nanovllm/engine/llm_engine.py) 驱动。

---

## 2. 整体架构概览

```
LLMEngine.generate()
    └── while not finished:
            step()
              ├── scheduler.schedule()          # 调度决策
              ├── model_runner.run(seqs, ...)    # 模型推理
              └── scheduler.postprocess()       # 结果写回，更新队列
```

三个关键模块分工：

| 模块 | 文件 | 职责 |
|------|------|------|
| `Scheduler` | [scheduler.py](../nanovllm/engine/scheduler.py) | 管理 waiting/running 队列，决定本次迭代处理哪些序列 |
| `ModelRunner` | [model_runner.py](../nanovllm/engine/model_runner.py) | 组装输入 Tensor，驱动模型前向计算 |
| `BlockManager` | [block_manager.py](../nanovllm/engine/block_manager.py) | 管理物理 KV Cache 块的分配/释放/前缀复用 |

---

## 3. 核心数据结构

### 3.1 Sequence

> 源码：[nanovllm/engine/sequence.py](../nanovllm/engine/sequence.py)

```python
class Sequence:
    block_size = 256          # 每个 KV Cache 物理块存多少 token

    def __init__(self, token_ids, sampling_params):
        self.token_ids = copy(token_ids)   # 完整 token 历史
        self.num_tokens = len(token_ids)   # 当前总 token 数（随 decode 递增）
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0         # prefix cache 命中的 token 数
        self.block_table = []              # 物理块 id 列表（Paged Attention）
        self.last_token = token_ids[-1]    # decode 阶段只需要最后一个 token
```

关键属性：

- `num_cached_tokens`：prefix cache 命中数，prefill 时可跳过这些 token 的计算。
- `block_table`：该序列的逻辑块→物理块映射，对应 vLLM 的 page table。
- `last_token`：decode 阶段每步只需这一个 token 参与前向计算。

`__getstate__` / `__setstate__` 用于 Tensor Parallelism 下通过 `pickle` 跨进程传递序列，prefill 传全部 token，decode 只传最后一个，节省 IPC 开销。

### 3.2 Block 与 BlockManager

> 源码：[nanovllm/engine/block_manager.py](../nanovllm/engine/block_manager.py)

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0    # 共享块的引用计数（prefix cache 共享）
        self.hash = -1        # 内容哈希，用于 prefix cache 查找
        self.token_ids = []   # 块内 token 内容（用于哈希碰撞校验）
```

`BlockManager` 维护：

- `free_block_ids`：空闲物理块队列（deque，FIFO 分配）
- `used_block_ids`：已使用的物理块集合
- `hash_to_block_id`：token 内容哈希 → 物理块 id（prefix cache 的核心索引）

物理 KV Cache 在 [`ModelRunner.allocate_kv_cache()`](../nanovllm/engine/model_runner.py#L120) 中一次性预分配：

```python
# shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers,
                            config.num_kvcache_blocks, self.block_size,
                            num_kv_heads, head_dim)
```

每层 Attention 模块的 `k_cache` / `v_cache` 直接指向这块连续内存的对应切片：

```python
module.k_cache = self.kv_cache[0, layer_id]  # shape: [num_blocks, block_size, H, D]
module.v_cache = self.kv_cache[1, layer_id]
```

### 3.3 Context 全局上下文

> 源码：[nanovllm/utils/context.py](../nanovllm/utils/context.py)

```python
@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q: torch.Tensor   # prefix-sum 序列长度，prefill 用
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: torch.Tensor   # 每个 token → KV Cache 写入槽位
    context_lens: torch.Tensor   # decode 用：每个序列当前总长度
    block_tables: torch.Tensor   # 每个序列的物理块表
```

`set_context()` 在每次迭代前由 `prepare_prefill` / `prepare_decode` 设置，模型内部所有层通过 `get_context()` 读取，**无需改动模型签名**，是一种全局 sidecar 的设计模式。

---

## 4. 调度器：Scheduler 全流程

> 源码：[nanovllm/engine/scheduler.py](../nanovllm/engine/scheduler.py)

### 4.1 核心队列设计

```python
self.waiting: deque[Sequence] = deque()   # 等待 prefill 的请求
self.running: deque[Sequence] = deque()   # 正在 decode 的请求
```

NanoVLLM 的关键设计决策：**Prefill 和 Decode 严格不混合**。每次 `schedule()` 返回一个 flag `is_prefill`，要么全部是 prefill 序列，要么全部是 decode 序列。  

> 需要参考VLLM实现，VLLM应该是支持Preill 和 Decode混合的。

### 4.2 Prefill 调度

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        if (num_batched_tokens + len(seq) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)          # 分配物理块并检测 prefix cache
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True               # is_prefill = True
```

调度约束：

1. `num_batched_tokens + len(seq) <= max_num_batched_tokens`：本次 batch 的总 token 数上限（默认 16384），防止单次前向 OOM。
2. `block_manager.can_allocate(seq)`：KV Cache 物理块是否够用。
3. **贪心策略**：按 FIFO 顺序尽量多地装入 prefill 序列。

`allocate` 内部执行 prefix cache 探测（见第 8 节），命中时将 `seq.num_cached_tokens` 置为已缓存 token 数，从而在 `num_batched_tokens` 计算中跳过已缓存部分：

```python
num_batched_tokens += len(seq) - seq.num_cached_tokens  # 只计算未缓存的 token
```

### 4.3 Decode 调度与抢占

当 `waiting` 为空或 prefill 批次容量耗尽，进入 decode 调度分支：

```python
while self.running and num_seqs < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())   # 抢占 running 末尾优先级最低的序列
        else:
            self.preempt(seq)                 # 连自身都放不下，直接抢占自己
            break
    else:
        num_seqs += 1
        self.block_manager.may_append(seq)    # 追加新 token 的 KV Cache slot
        scheduled_seqs.append(seq)
```

`can_append` 判断：当前 token 是否需要新开一个物理块（`len(seq) % block_size == 1` 时才需要新块）；`may_append` 完成实际块分配与哈希表更新。

**抢占**（Preemption）逻辑：

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)   # 释放物理块，KV Cache 丢弃
    self.waiting.appendleft(seq)         # 重新排回 waiting 队列头部
```

被抢占的序列释放 KV Cache 后重新进入 waiting 队列，下次再做完整 prefill（NanoVLLM 是 **Recompute** 策略，不做 Swap 到 CPU）。

### 4.4 后处理 postprocess

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) \
                or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

每步迭代后将新生成的 token 写入序列，判断终止条件（EOS 或达到 max_tokens），完成的序列立刻释放 KV Cache 块，**这就是 continuous batching 中"slot 即时回收"的关键**：新请求随时可以占用刚释放的物理块。

---

## 5. ModelRunner：Batch 组装

> 源码：[nanovllm/engine/model_runner.py](../nanovllm/engine/model_runner.py)

`run()` 是推理入口：

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    input_ids, positions = (self.prepare_prefill(seqs) if is_prefill
                            else self.prepare_decode(seqs))
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

### 5.1 prepare\_prefill

将多个序列的 token 拼接成**一个大的 1D 张量**（Packed / Variable-length 格式）：

```python
def prepare_prefill(self, seqs):
    input_ids, positions = [], []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    slot_mapping = []
    for seq in seqs:
        seqlen = len(seq)
        # 只加入未命中 prefix cache 的 token
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(range(seq.num_cached_tokens, seqlen))
        seqlen_q = seqlen - seq.num_cached_tokens
        seqlen_k = seqlen
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        # 计算每个 token 写入哪个 KV Cache slot
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            end = (start + self.block_size
                   if i != seq.num_blocks - 1
                   else start + seq.last_block_num_tokens)
            slot_mapping.extend(range(start, end))
    ...
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
    return input_ids, positions
```

以 3 个序列 `[S1=4tok, S2=6tok, S3=3tok]` 为例，假设均无 prefix cache 命中：

```
input_ids  = [t1_0, t1_1, t1_2, t1_3,  t2_0, ..., t2_5,  t3_0, t3_1, t3_2]
              |------ seq1 ------|        |------- seq2 ------|   |-- seq3 --|
positions  = [0, 1, 2, 3,                0, 1, 2, 3, 4, 5,       0, 1, 2]
cu_seqlens_q = [0, 4, 10, 13]    # prefix-sum of query lengths
cu_seqlens_k = [0, 4, 10, 13]    # prefix-sum of key lengths（无 prefix cache 时同上）
```

`cu_seqlens_q` 和 `cu_seqlens_k` 是 FlashAttention varlen 接口的关键参数，告诉内核**在哪里切分序列边界**，保证注意力掩码不会跨序列泄露。

### 5.2 prepare\_decode

Decode 阶段每个序列只送最后一个 token：

```python
def prepare_decode(self, seqs):
    input_ids, positions, slot_mapping, context_lens = [], [], [], []
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        # KV 写入槽位：最后一个物理块的当前偏移
        slot_mapping.append(seq.block_table[-1] * self.block_size
                             + seq.last_block_num_tokens - 1)
    ...
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

对于 batch 中 B 个序列，decode 阶段：

```
input_ids  shape: [B]        每个序列最新生成的 token
positions  shape: [B]        每个序列当前位置（len(seq)-1）
context_lens shape: [B]      每个序列的总历史长度，告诉 FlashAttn 读多少 KV
block_tables shape: [B, max_blocks]  每个序列的 page table
```

---

## 6. Batch 如何喂给模型

> 源码：[nanovllm/models/qwen3.py](../nanovllm/models/qwen3.py)，[nanovllm/layers/attention.py](../nanovllm/layers/attention.py)

### 6.1 Embedding 层

```python
# Qwen3Model.forward
hidden_states = self.embed_tokens(input_ids)  # [total_tokens, hidden_size]
```

无论是 prefill 还是 decode，`input_ids` 都是一个 1D 张量，embedding 后得到 `[total_tokens, hidden_size]` 的 2D 张量。**所有序列的 token 均在 dim=0 上顺序拼接，没有 padding**。

### 6.2 MLP 层 —— 无需感知 Batch 边界

```python
# Qwen3MLP.forward
gate_up = self.gate_up_proj(x)   # [total_tokens, 2 * intermediate_size]
x = self.act_fn(gate_up)         # SiluAndMul，逐 token 操作
x = self.down_proj(x)            # [total_tokens, hidden_size]
```

MLP 是逐 token 的 point-wise 操作（线性映射 + 激活），**天然支持任意 total_tokens 的 batch 拼接，无需关心序列边界**。这是 Continuous Batching 中 MLP 部分的优雅之处：来自不同序列的 token 直接拼在一起做矩阵乘，没有任何额外开销。

### 6.3 Attention 层 —— Prefill 阶段

> 源码：[nanovllm/layers/attention.py#L128](../nanovllm/layers/attention.py#L128)

```python
# Qwen3Attention.forward
qkv = self.qkv_proj(hidden_states)               # [total_tokens, (H_q+2*H_kv)*D]
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
q = q.view(-1, num_heads, head_dim)              # [total_tokens, H_q, D]
k = k.view(-1, num_kv_heads, head_dim)           # [total_tokens, H_kv, D]
v = v.view(-1, num_kv_heads, head_dim)
q, k = self.rotary_emb(positions, q, k)          # RoPE 使用 positions 向量

# 进入 Attention.forward
o = self.attn(q, k, v)
```

在 `Attention.forward` 中：

```python
# 1. 写入 KV Cache（slot_mapping 告诉每个 token 写哪个物理槽）
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

# 2. 调用 FlashAttention varlen 接口
o = flash_attn_varlen_func(
    q,                             # [total_q_tokens, H, D]
    k,                             # [total_k_tokens, H_kv, D]
    v,
    cu_seqlens_q=context.cu_seqlens_q,   # [batch+1]，序列边界
    cu_seqlens_k=context.cu_seqlens_k,   # [batch+1]
    max_seqlen_q=context.max_seqlen_q,
    max_seqlen_k=context.max_seqlen_k,
    softmax_scale=self.scale,
    causal=True,
    block_table=context.block_tables  # None 或 prefix cache 的 page table
)
```

`flash_attn_varlen_func` 是 FlashAttention2 提供的**变长序列批处理接口**，其核心原理：

- 输入是多个序列 token 拼接的长向量（没有 padding）。
- `cu_seqlens_q`（cumulative sequence lengths）标注每个序列的起止偏移，例如 `[0, 4, 10, 13]` 代表第 1 个序列是 `[0:4]`，第 2 个是 `[4:10]`，第 3 个是 `[10:13]`。
- FlashAttention 内核保证 Attention 计算**不跨越序列边界**，每个序列只看自己的 K/V，并应用 causal mask，完全等价于对每个序列单独做 Attention，但 GPU 利用率大幅提升（消除了 padding 带来的无效计算）。

**数据流示意（3 序列 prefill）：**

```
q:  [q1_0, q1_1, q1_2, q1_3 | q2_0, q2_1, q2_2, q2_3, q2_4, q2_5 | q3_0, q3_1, q3_2]
     ←—————— seq1 ——————————→  ←——————————— seq2 ——————————————————→  ←—— seq3 ——————→
     (causal attn: 每行只 attend 当前及之前的同序列 token)

cu_seqlens_q = [0, 4, 10, 13]   →  三个独立的因果注意力窗口
```

### 6.4 Attention 层 —— Decode 阶段

```python
# Attention.forward decode 分支
o = flash_attn_with_kvcache(
    q.unsqueeze(1),             # [B, 1, H, D]  每个序列只有 1 个 query token
    k_cache,                    # [num_blocks, block_size, H_kv, D]
    v_cache,
    cache_seqlens=context.context_lens,   # [B]  每个序列的历史 KV 长度
    block_table=context.block_tables,     # [B, max_blocks]
    softmax_scale=self.scale,
    causal=True
)
```

`flash_attn_with_kvcache` 是 FlashAttention 专为 auto-regressive decode 设计的接口，关键点：

- 每个序列的查询只有 1 个 token（新生成的），但 K/V 来自整个历史（存在物理块中）。
- `block_table` 是一个 `[B, max_blocks]` 的页表，FlashAttention 内核按页表读取分散的物理块组成逻辑连续的 KV 序列。
- `cache_seqlens` 告诉内核每个序列的有效 KV 长度，超出部分忽略。

**数据流示意（B=3 序列 decode）：**

```
q:  [q1_new | q2_new | q3_new]    shape: [3, 1, H, D]

k_cache / v_cache (物理连续的分页存储):
  Block 0: [tok0~255 of seq2]
  Block 1: [tok0~255 of seq1]
  Block 3: [tok256~511 of seq2]
  ...

block_tables:
  seq1: [1, 5, ...]     → 按此顺序读取物理块，拼成 seq1 的完整 KV 历史
  seq2: [0, 3, ...]
  seq3: [2, ...]

context_lens = [512, 768, 256]    → 分别读取前 512/768/256 个有效 KV
```

---

## 7. KV Cache 写入：store\_kvcache Triton Kernel

> 源码：[nanovllm/layers/attention.py#L1](../nanovllm/layers/attention.py#L1)

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)          # 第 idx 个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return           # padding token，跳过
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

**slot_mapping 的含义：**

```
slot = block_table[block_idx] * block_size + token_offset_in_block
```

例如 seq1 的第 260 个 token（超过第一个 block 的 256 个），其 slot：

```
block_idx = 260 // 256 = 1
offset    = 260 % 256  = 4
block_id  = seq.block_table[1]   (假设为物理块 7)
slot      = 7 * 256 + 4 = 1796
```

Triton kernel 以 `N`（本次 forward 的 token 总数）个并行 program 运行，每个 program 负责一个 token 的 KV 写入，天然适配拼接后的大 batch。

---

## 8. Paged Attention + Prefix Cache

> 源码：[nanovllm/engine/block_manager.py](../nanovllm/engine/block_manager.py)

### 块分配与哈希

```python
def allocate(self, seq: Sequence):
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # 只有整块才能计算确定性哈希（最后一块可能未满，内容还会变化）
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            seq.num_cached_tokens += self.block_size  # prefix cache 命中！
            block.ref_count += 1
        seq.block_table.append(block_id)
```

哈希依赖链：$h_i = \text{hash}(\text{tokens}_i, h_{i-1})$，类似 Merkle 树，确保完整路径上的 prefix 才能命中缓存，防止哈希碰撞导致错误的 KV 复用。

### Decode 阶段追加块

```python
def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        # 当前最新 token 是新块的第一个，分配新物理块
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        seq.block_table.append(block_id)
    elif len(seq) % self.block_size == 0:
        # 当前块刚好填满，更新哈希表，为未来的 prefix cache 命中做准备
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
```

---

## 9. CUDAGraph 加速 Decode

> 源码：[nanovllm/engine/model_runner.py#L245](../nanovllm/engine/model_runner.py#L245)

Decode 阶段 batch size 较小且形状固定（每个序列 1 个 token），非常适合 CUDAGraph 录制：

```python
# 录制多个 batch size 档位的 CUDAGraph
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    self.graphs[bs] = graph
```

推理时按序号找满足条件的最小档位：

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
# 填充 placeholder tensors
graph_vars["input_ids"][:bs] = input_ids
...
graph.replay()   # 零 CPU 开销的 GPU kernel 重放
```

**Prefill 不使用 CUDAGraph**（输入形状每次不同），直接 eager 执行。

---

## 10. Tensor Parallelism 中的 Batch 拼接

> 源码：[nanovllm/models/qwen3.py](../nanovllm/models/qwen3.py)，[nanovllm/layers/linear.py](../nanovllm/layers/linear.py)

Tensor Parallel 模式下，QKV 投影按 head 维度列切分，O 投影行切分：

```
QKVParallelLinear:  [hidden_size] → [(H_q/tp + 2*H_kv/tp) * D]  每卡持有 1/tp 的 heads
RowParallelLinear:  [(H_total * D)] → [hidden_size]              AllReduce 合并

MergedColumnParallelLinear: gate/up 列切分
RowParallelLinear:           down 行切分
```

对 batch 拼接而言，TP 完全透明：输入和输出的 `total_tokens` 维度不变，只有 hidden / head 维度被切分，Continuous Batching 的 token 拼接逻辑无需修改。

跨 rank 通信通过共享内存 (SharedMemory) + Event 实现低延迟 IPC：

```python
# rank 0 写入 shm，通知其他 rank
self.write_shm("run", seqs, is_prefill)
# rank 1..N 在 loop() 中等待 Event，读取 shm 后执行相同的 run()
```

---

## 11. 端到端数据流图

```
用户请求 prompt → LLMEngine.add_request()
    ↓  tokenize
    Sequence(token_ids) → scheduler.waiting

────── 每次迭代 step() ──────────────────────────────────────────────────
                                                                         
  scheduler.schedule()                                                   
  ┌──────────────────────────────────────────────────────────────────┐   
  │  waiting 非空?  →  prefill 调度                                   │   
  │    检查 max_batched_tokens & can_allocate                         │   
  │    BlockManager.allocate() 分配物理块 & 检测 prefix cache 命中    │   
  │    移入 running, scheduled_seqs, is_prefill=True                 │   
  │                                                                  │   
  │  否则 → decode 调度                                               │   
  │    遍历 running, 检查 can_append                                  │   
  │    不够 → 抢占末尾序列 → 释放 KV Cache → 送回 waiting 头          │   
  │    BlockManager.may_append() 追加 slot                           │   
  │    is_prefill=False                                              │   
  └──────────────────────────────────────────────────────────────────┘   
                    │ scheduled_seqs, is_prefill                         
                    ↓                                                     
  ModelRunner.run(seqs, is_prefill)                                       
  ┌──────────────────────────────────────────────────────────────────┐   
  │  prepare_prefill / prepare_decode                                │   
  │    拼接 input_ids (1D, no padding)                               │   
  │    计算 positions                                                 │   
  │    计算 slot_mapping (每 token → KV 物理槽)                       │   
  │    计算 cu_seqlens_q/k (prefill) 或 context_lens/block_tables    │   
  │    set_context(...)  →  全局 sidecar 传给所有 Attention 层        │   
  │                                                                  │   
  │  model(input_ids, positions)                                     │   
  │    embed_tokens → [total_tokens, hidden]                         │   
  │    for layer in layers:                                          │   
  │      RMSNorm → Attention → RMSNorm → MLP                        │   
  │      ┌─ Attention ─────────────────────────────────────────┐    │   
  │      │  qkv_proj [total_tokens, hidden]→[total_tokens, qkv] │    │   
  │      │  rotary_emb (position-wise)                          │    │   
  │      │  store_kvcache (Triton, slot_mapping)                │    │   
  │      │  prefill: flash_attn_varlen_func (cu_seqlens)        │    │   
  │      │  decode:  flash_attn_with_kvcache (block_tables)     │    │   
  │      │  o_proj (RowParallel + AllReduce if TP)              │    │   
  │      └──────────────────────────────────────────────────────┘    │   
  │      MLP: gate_up_proj → SiluAndMul → down_proj (all point-wise) │   
  │                                                                  │   
  │  compute_logits → sampler (top-k / temperature)                 │   
  │  return token_ids                                                │   
  └──────────────────────────────────────────────────────────────────┘   
                    │ token_ids                                           
                    ↓                                                     
  scheduler.postprocess()                                                 
    seq.append_token(token_id)                                           
    is_finished? → deallocate KV, 移出 running, 输出结果               
────── 继续下一次迭代 ────────────────────────────────────────────────────
```

---

## 关键设计总结

| 问题 | NanoVLLM 的解决方案 |
|------|-------------------|
| 不同长度序列如何同时处理 Attention？ | `flash_attn_varlen_func` + `cu_seqlens` 无 padding 拼接 |
| Decode 阶段如何高效读取历史 KV？ | Paged KV Cache + `flash_attn_with_kvcache` 按 block_table 读 |
| 如何感知序列边界而不改模型签名？ | 全局 `Context` sidecar，`set_context` / `get_context` |
| 如何避免 KV Cache 内存碎片？ | Paged Attention，固定大小的物理块（默认 256 token/块） |
| 如何利用重复 prompt 节省计算？ | Prefix Cache，基于哈希链的块共享与引用计数 |
| MLP 如何处理 batch 拼接？ | MLP 是 token-wise 操作，天然支持任意拼接，无需额外处理 |
| Decode 小 batch 如何减少 CPU 开销？ | CUDAGraph 按 batch size 档位录制，replay 无 CPU 开销 |
| 多 GPU 如何协同？ | SharedMemory + Event IPC，rank 0 广播，各 rank 执行相同计算，AllReduce 聚合 |

---

*文档生成时间：2026-02-26。对应代码位置均以相对路径链接，可在 VS Code 中直接跳转。*
