# NanoVLLM KV Cache 深度解读

本文档详细梳理 NanoVLLM 中 KV Cache 的完整实现，涵盖 **Paged Attention**、**Prefix Cache** 等核心技术，从预开辟、Prefill/Decode 差异化准备到推理算子的实际操作。

> 本文完全由Claude Sonnet4.6生成

---

## 一、整体架构概览

NanoVLLM 的 KV Cache 系统涉及以下核心文件：

| 文件 | 职责 |
| :--- | :--- |
| `engine/block_manager.py` | 页式 KV Cache 的块分配、复用、回收，Prefix Cache 哈希匹配 |
| `engine/sequence.py` | 序列数据结构，维护 `block_table`（逻辑→物理块映射） |
| `engine/scheduler.py` | 调度器，决定 Prefill/Decode 及 block 分配/追加时机 |
| `engine/model_runner.py` | 预开辟 KV Cache、构造 `slot_mapping`/`block_tables` 等推理输入 |
| `layers/attention.py` | Triton kernel 写入 KV Cache + Flash Attention 分页读取 |
| `utils/context.py` | 全局上下文，传递 `slot_mapping`、`block_tables` 等信息到 Attention 层 |

---

## 二、KV Cache 预开辟（初始化阶段）

### 2.1 流程

在 `ModelRunner.__init__` 中按顺序执行：

```
ModelRunner.__init__()
  → warmup_model()         # Step 1: 探测 peak memory
  → allocate_kv_cache()    # Step 2: 一次性分配全部 KV Cache
  → capture_cudagraph()    # Step 3: 为 decode 录制 CUDAGraph
```

### 2.2 Step 1: Warmup 探测 peak memory

```python
# model_runner.py - warmup_model()
seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
self.run(seqs, True)
```

用全 0 序列跑一次前向传播，让 PyTorch 记录 `torch.cuda.memory_stats()["allocated_bytes.all.peak"]`，为后续计算可分配块数提供依据。

### 2.3 Step 2: 一次性分配全部 KV Cache

```python
# model_runner.py - allocate_kv_cache()

# 计算每个 block 的字节大小
block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize

# 用剩余显存算出能分多少个 block
num_kvcache_blocks = (total * gpu_memory_utilization - used - peak + current) // block_bytes

# 核心：一次 malloc，6 维连续张量
self.kv_cache = torch.empty(2, L, B, T, H, D)
```

6 维张量各维度含义：

| 维度 | 符号 | 含义 |
| :--- | :--- | :--- |
| dim 0 | `2` | K 和 V 两份 cache |
| dim 1 | `L` | `num_hidden_layers`，模型层数 |
| dim 2 | `B` | `num_kvcache_blocks`，物理块数量 |
| dim 3 | `T` | `block_size`（默认 256），每块可存的 token 数 |
| dim 4 | `H` | `num_kv_heads`（per rank），KV 头数 |
| dim 5 | `D` | `head_dim`，每个头的维度 |

以 Qwen3-8B（`L=32, num_kv_heads=8, head_dim=128, block_size=256, bf16`）、单卡为例：

$$\text{block\_bytes} = 2 \times 32 \times 256 \times 8 \times 128 \times 2 = 268\text{MB/block}$$

### 2.4 Step 3: 绑定到每层 Attention

```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]  # shape: [B, T, H, D]
        module.v_cache = self.kv_cache[1, layer_id]  # shape: [B, T, H, D]
        layer_id += 1
```

**关键设计**：预分配一整块连续显存，避免推理过程中反复 malloc/free 导致碎片化和延迟。每层 Attention 只是持有该大张量的一个 view，零拷贝。

---

## 三、Paged Attention — BlockManager 块管理

### 3.1 核心数据结构

```python
class Block:
    block_id: int           # 物理块 ID
    ref_count: int          # 引用计数（Prefix Cache 共享）
    hash: int               # 链式哈希值（-1 表示不可缓存）
    token_ids: list[int]    # 块内 token 内容（用于哈希校验）

class BlockManager:
    blocks: list[Block]                    # 所有物理块元数据
    free_block_ids: deque[int]             # 空闲块队列
    used_block_ids: set[int]               # 已使用块集合
    hash_to_block_id: dict[int, int]       # hash → block_id（Prefix Cache 核心）
```

每个 `Sequence` 维护一个 `block_table: list[int]`，存储该序列的**逻辑块→物理块 ID** 的映射。

### 3.2 逻辑→物理寻址

物理 KV 存储的地址计算：

$$\text{slot} = \text{block\_table}[i] \times \text{block\_size} + \text{offset}$$

这就是 **Paged Attention** 的核心：序列的 KV 在物理上不需要连续，通过 `block_table` 间接寻址，类似操作系统的页表机制。

```
逻辑视图（连续）         物理视图（分散）
┌────────┐              ┌────────┐ block 5
│ Block 0 │ ──────────→ │ tokens │
├────────┤              └────────┘
│ Block 1 │ ──────┐     ┌────────┐ block 2
├────────┤       └───→ │ tokens │
│ Block 2 │ ──────┐     └────────┘
└────────┘       │     ┌────────┐ block 9
                  └───→ │ tokens │
                        └────────┘
```

### 3.3 Prefill 阶段: `allocate(seq)`

遍历序列的每个逻辑 block：

1. **满块**（`len(token_ids) == block_size`）→ 计算 `xxhash(token_ids, prefix_hash)`
2. **哈希命中且 token_ids 一致** → **Prefix Cache HIT**，直接复用，`ref_count++`，`num_cached_tokens += block_size`
3. **未命中或末尾不满** → 从 `free_block_ids` 分配新物理块
4. 所有分配的 `block_id` 追加到 `seq.block_table`

```python
def allocate(self, seq: Sequence):
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # 只有满块才计算 hash
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        if cache_miss:
            block = self._allocate_block(self.free_block_ids[0])
        else:
            seq.num_cached_tokens += self.block_size
            ...
        seq.block_table.append(block_id)
```

### 3.4 Decode 阶段: `may_append(seq)`

每生成一个新 token 后调用，按三种情况处理：

| 条件 | 操作 |
| :--- | :--- |
| `len(seq) % block_size == 1` | 上一个 block 刚满，分配新 block |
| `len(seq) % block_size == 0` | 当前 block 刚好填满，计算 hash 注册缓存 |
| 其他 | 当前 block 还有空位，无需操作 |

### 3.5 释放: `deallocate(seq)`

逆序释放 `block_table` 中的物理块，`ref_count--`，降为 0 时才真正回收到 `free_block_ids`（支持 Prefix Cache 共享）。

---

## 四、Prefix Cache 机制

### 4.1 工作原理

Prefix Cache 让 **不同请求共享相同前缀的 KV Cache**（如 system prompt），避免重复计算。

#### 链式哈希

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 前一个 block 的 hash
    h.update(np.array(token_ids).tobytes())       # 当前 block 的 token_ids
    return h.intdigest()
```

每个 block 的 hash = f(前缀 hash, 当前 block 的 token_ids)。这保证了只有当 **完整前缀 token 序列完全一致** 时 hash 才会匹配，杜绝了不同上下文下的碰撞。

#### 安全校验

- 仅对 **满块** 计算 hash（未满的 block 内容还会变化）
- 命中后额外校验 `block.token_ids != token_ids`，防止哈希碰撞

#### 引用计数共享

命中后 `ref_count++`，多个序列可以共享同一个物理 block。只有 `ref_count == 0` 时才回收。

### 4.2 Prefix Cache 示例

```
请求 A: "你是一个AI助手。请回答：什么是深度学习？"
请求 B: "你是一个AI助手。请回答：什么是强化学习？"

┌─────────────────────┐        ┌─────────────────┐
│  "你是一个AI助手。"  │ ← 共享   │ "请回答：什么是  │ ← 请求 A 独有
│  Block 0 (hash=0xA) │        │ 深度学习？"      │
│  ref_count = 2      │        │ Block 1          │
└─────────────────────┘        └─────────────────┘
         ↑                     ┌─────────────────┐
         └──── 共享 ──────────  │ "请回答：什么是  │ ← 请求 B 独有
                               │ 强化学习？"      │
                               │ Block 2          │
                               └─────────────────┘
```

请求 B 到来时，Block 0 的 hash 命中且 token_ids 一致，直接复用，跳过这部分的 KV 计算。

---

## 五、Prefill 与 Decode 的准备差异

### 5.1 `prepare_prefill()` — 多 token 输入

```python
def prepare_prefill(self, seqs: list[Sequence]):
    for seq in seqs:
        input_ids.extend(seq[seq.num_cached_tokens:])        # 跳过已缓存的 token
        positions.extend(range(seq.num_cached_tokens, seqlen))
        cu_seqlens_q.append(... + seqlen_q)                  # Q 长度 = 未缓存 token 数
        cu_seqlens_k.append(... + seqlen_k)                  # K 长度 = 全部上下文
        # slot_mapping 从 num_cached_blocks 开始，只为未缓存的 token 生成写入地址
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            slot_mapping.extend(range(start, end))
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:                  # Q < K → 有 prefix cache
        block_tables = self.prepare_block_tables(seqs)
```

各字段含义：

| 字段 | Prefill 时的值 | 说明 |
| :--- | :--- | :--- |
| `input_ids` | `seq[num_cached_tokens:]` | **跳过已缓存的 token**，减少计算量 |
| `positions` | `range(num_cached_tokens, seqlen)` | 位置编码也跳过已缓存部分 |
| `cu_seqlens_q` | 累加**未缓存长度** | Q 的实际计算长度 |
| `cu_seqlens_k` | 累加**全部长度** | K 包含已缓存 + 新计算的所有 token |
| `slot_mapping` | 多个 slot（从 cached_blocks 开始） | 只为未缓存 token 生成写入地址 |
| `block_tables` | 有 prefix cache 时非 None | 用于 flash_attn 从分页 cache 读取 |

**Prefix Cache 的识别**：当 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`（K 长度 > Q 长度），说明部分 token 已在 cache 中，此时需要构造 `block_tables`。

### 5.2 `prepare_decode()` — 单 token 输入

```python
def prepare_decode(self, seqs: list[Sequence]):
    for seq in seqs:
        input_ids.append(seq.last_token)                     # 每个 seq 只有 1 个新 token
        positions.append(len(seq) - 1)
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )
        context_lens.append(len(seq))                        # 完整上下文长度
    block_tables = self.prepare_block_tables(seqs)           # 始终需要
```

| 字段 | Decode 时的值 | 说明 |
| :--- | :--- | :--- |
| `input_ids` | `[last_token]` | 每个 seq 只有 **1 个新 token** |
| `positions` | `[len(seq) - 1]` | 最后一个位置 |
| `slot_mapping` | 1 个 slot/seq | 当前 token 要写入的唯一物理位置 |
| `context_lens` | `[len(seq)]` | 告诉 flash_attn 每个 seq 的 KV 总长度 |
| `block_tables` | **始终构造** | decode 必须从 cache 读取全部历史 KV |

### 5.3 核心差异对比

```
Prefill:
  ┌──────────────────────────────────────────────┐
  │ 多 token 输入                                 │
  │ 可能跳过 prefix（num_cached_tokens > 0）       │
  │ slot_mapping 对应多个 slot                     │
  │ Q/K 长度可能不等（有 prefix cache 时 Q < K）    │
  │ block_tables 可能为 None（无 prefix cache）     │
  └──────────────────────────────────────────────┘

Decode:
  ┌──────────────────────────────────────────────┐
  │ 1 token 输入                                  │
  │ 写 1 个 slot                                   │
  │ 从 block_table 读全部历史 KV                   │
  │ Q = 1, K = 全部上下文                          │
  │ block_tables 始终非 None                       │
  └──────────────────────────────────────────────┘
```

---

## 六、推理算子如何操作 KV Cache

核心在 `layers/attention.py` 的 `Attention.forward()` 方法中。

### 6.1 写入 KV Cache — Triton Kernel `store_kvcache`

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

#### Triton Kernel 工作方式

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)                              # 每个 token 一个 program
    slot = tl.load(slot_mapping_ptr + idx)               # 读取物理槽位
    if slot == -1: return                                # CUDAGraph padding 跳过

    key_offsets = idx * key_stride + tl.arange(0, D)
    cache_offsets = slot * D + tl.arange(0, D)           # D = num_kv_heads * head_dim

    key = tl.load(key_ptr + key_offsets)
    tl.store(k_cache_ptr + cache_offsets, key)           # 写入 k_cache
    value = tl.load(value_ptr + value_offsets)
    tl.store(v_cache_ptr + cache_offsets, value)         # 写入 v_cache
```

核心原理：
- **N 个 program 并行**（每个 token 一个 program）
- 每个 program 读取 `slot_mapping[idx]` 得到物理槽位
- `D = num_kv_heads * head_dim`，将 KV 的 `[num_kv_heads, head_dim]` 拉平为一维
- `cache_offsets = slot * D + arange(0, D)`，直接通过指针算术定位写入位置

#### 内存布局示意

```
k_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
                 │            │            │             │
                 │            │            └─── D ───────┘  (连续)
                 │            │
                 │            └── slot = block_id * block_size + offset
                 └── 物理块编号

slot_mapping[i] = block_table[logical_block] * block_size + token_offset_in_block
```

### 6.2 读取 KV Cache — Flash Attention

根据阶段和是否有 Prefix Cache，共有三条路径：

#### 路径 1: Prefill 无 Prefix Cache

```python
o = flash_attn_varlen_func(q, k, v,
                           cu_seqlens_q=..., cu_seqlens_k=...,
                           softmax_scale=scale, causal=True,
                           block_table=None)
```

直接用当前 batch 计算得到的稠密 k, v 做 attention（KV 已写入 cache 供后续 decode 使用，但本次 attention 不需要读 cache）。

#### 路径 2: Prefill 有 Prefix Cache

```python
k, v = k_cache, v_cache   # 替换为完整 cache
o = flash_attn_varlen_func(q, k_cache, v_cache,
                           cu_seqlens_q=..., cu_seqlens_k=...,
                           softmax_scale=scale, causal=True,
                           block_table=block_tables)
```

Q 只有未缓存的 token，但需要 attend 到全部 K（含已缓存部分），通过 `block_table` 从分页 cache 读取。

#### 路径 3: Decode

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),         # [batch, 1, num_heads, head_dim]
    k_cache, v_cache,       # [num_blocks, block_size, num_kv_heads, head_dim]
    cache_seqlens=context_lens,
    block_table=block_tables,
    softmax_scale=scale, causal=True
)
```

每个序列只有 1 个 Q token，从分页 cache 中按 `block_table` 读取全部历史 KV 做 attention。`cache_seqlens` 告诉 flash_attn 每个序列实际有多少有效 KV。

### 6.3 三条路径对比

| | Prefill (无 prefix) | Prefill (有 prefix) | Decode |
| :--- | :--- | :--- | :--- |
| **Q shape** | `[total_q, H, D]` | `[total_q, H, D]`（仅未缓存） | `[batch, 1, H, D]` |
| **K/V 来源** | 当前 batch 的 k, v | `k_cache`, `v_cache` | `k_cache`, `v_cache` |
| **block_table** | `None` | 非 None | 非 None |
| **API** | `flash_attn_varlen_func` | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| **store_kvcache** | 写入所有 token | 写入未缓存 token | 写入 1 个 token |

---

## 七、Context 全局上下文传递

`utils/context.py` 定义了一个全局 `Context` dataclass，用于在 `prepare_*` 和 `Attention.forward` 之间传递分页信息：

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None     # Q 的累积序列长度
    cu_seqlens_k: torch.Tensor | None = None     # K 的累积序列长度
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None     # 写入地址
    context_lens: torch.Tensor | None = None     # decode 时的上下文长度
    block_tables: torch.Tensor | None = None     # 分页块表
```

数据流：`prepare_prefill/decode() → set_context() → Attention.forward() 中 get_context() → reset_context()`

---

## 八、完整流程图

```
                             ┌─────────────────────────────────────────┐
                             │     1. 初始化阶段 — 预开辟              │
                             │                                         │
                             │  ModelRunner.__init__()                 │
                             │     │                                   │
                             │     ▼                                   │
                             │  warmup_model()                        │
                             │  用全0序列跑一次前向，触发 peak memory   │
                             │     │                                   │
                             │     ▼                                   │
                             │  allocate_kv_cache()                   │
                             │  计算可用显存 → num_kvcache_blocks      │
                             │     │                                   │
                             │     ▼                                   │
                             │  torch.empty(2, L, B, T, H, D)        │
                             │  一次性分配全部 KV Cache                │
                             │     │                                   │
                             │     ▼                                   │
                             │  遍历模型每层 Attention                 │
                             │  module.k_cache = kv_cache[0, layer]   │
                             │  module.v_cache = kv_cache[1, layer]   │
                             │     │                                   │
                             │     ▼                                   │
                             │  capture_cudagraph()                   │
                             │  为 decode 阶段录制 CUDAGraph           │
                             └──────────────┬──────────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │     2. 调度阶段 — Scheduler.schedule()        │
                    │                                               │
                    │      waiting 非空?                            │
                    │      ├── Yes → Prefill 路径                   │
                    │      │         block_manager.allocate(seq)    │
                    │      │         分配 blocks + prefix cache     │
                    │      │         返回 is_prefill=True           │
                    │      │                                        │
                    │      └── No  → Decode 路径                    │
                    │                block_manager.may_append(seq)  │
                    │                按需追加新 block                │
                    │                返回 is_prefill=False          │
                    └──────────┬────────────────┬───────────────────┘
                               │                │
                     ┌─────────▼──────┐  ┌──────▼──────────┐
                     │ 3a. Prefill    │  │ 3b. Decode      │
                     │ 准备           │  │ 准备             │
                     │                │  │                  │
                     │ input_ids =    │  │ input_ids =     │
                     │  跳过cached    │  │  [last_token]   │
                     │                │  │                  │
                     │ slot_mapping = │  │ slot_mapping =  │
                     │  多个 slot     │  │  1 个 slot      │
                     │                │  │                  │
                     │ Q长度 ≤ K长度  │  │ context_lens =  │
                     │ (有prefix时<)  │  │  全部上下文长度  │
                     │                │  │                  │
                     │ block_tables:  │  │ block_tables:   │
                     │ 有prefix→非None│  │  始终非None     │
                     └────────┬───────┘  └───────┬─────────┘
                              │                  │
                              └────────┬─────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────┐
                │     4. Attention.forward — KV Cache 读写      │
                │                                              │
                │  ① store_kvcache (Triton kernel)             │
                │     N 个 program 并行，每个写 1 个 token 的 KV │
                │     slot = slot_mapping[idx]                 │
                │     k_cache[slot] ← key[idx]                 │
                │     v_cache[slot] ← value[idx]               │
                │                                              │
                │  ② Flash Attention 计算                      │
                │     ├── Prefill 无 prefix:                    │
                │     │   flash_attn_varlen_func(q, k, v)      │
                │     │   block_table=None                     │
                │     │                                        │
                │     ├── Prefill 有 prefix:                    │
                │     │   flash_attn_varlen_func(q, cache, cache)│
                │     │   block_table=block_tables              │
                │     │                                        │
                │     └── Decode:                               │
                │         flash_attn_with_kvcache(q, cache)     │
                │         cache_seqlens + block_table            │
                └──────────────────────────────────────────────┘
```

---

## 九、关键设计总结

### 9.1 为什么需要 Paged Attention？

| 问题 | 传统方案 | Paged Attention |
| :--- | :--- | :--- |
| 显存浪费 | 预分配 max_seq_len，实际利用率低 | 按需分配 block，接近 100% 利用率 |
| 内存碎片 | 连续分配，无法复用碎片 | 类似 OS 页表，离散 block 可灵活调度 |
| 前缀共享 | 无法共享 | 通过 hash 匹配 + ref_count 共享 |

### 9.2 Prefix Cache 的收益

- **计算节省**：跳过已缓存 token 的前向传播（`input_ids = seq[num_cached_tokens:]`）
- **显存节省**：共享的 block 只存一份（`ref_count` 机制）
- **典型场景**：多轮对话中的 system prompt、batch 请求中的共同前缀

### 9.3 重要配置参数

| 参数 | 默认值 | 影响 |
| :--- | :--- | :--- |
| `kvcache_block_size` | 256 | 块粒度，影响缓存命中率和元数据开销 |
| `gpu_memory_utilization` | 0.9 | KV Cache 块数量，太高容易 OOM |
| `max_num_seqs` | 512 | 并发度上限，影响 Decode 吞吐 |
| `max_num_batched_tokens` | 16384 | Prefill 吞吐上限 |
