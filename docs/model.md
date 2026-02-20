# Qwen3-0.6B 模型架构详解 × NanoVLLM 实现梳理

> 完全claude sonnet4.6 生成

---

## 目录

1. [模型整体结构](#一模型整体结构)
2. [Qwen3-0.6B 超参总览](#二qwen3-06b-超参总览)
3. [逐层详解](#三逐层详解)
   - [3.1 Token Embedding](#31-token-embedding--vocabparallelembedding)
   - [3.2 RMSNorm](#32-rmsnorm)
   - [3.3 Grouped-Query Attention (GQA)](#33-grouped-query-attention-gqa)
   - [3.4 Per-Head QK Norm](#34-per-head-qk-norm)
   - [3.5 RoPE 旋转位置编码](#35-rope-旋转位置编码)
   - [3.6 Paged KV Cache + Flash Attention](#36-paged-kv-cache--flash-attention)
   - [3.7 SwiGLU MLP](#37-swiglu-mlp)
   - [3.8 LM Head + 采样](#38-lm-head--采样)
4. [关键实现细节与注意事项](#五关键实现细节与注意事项)
5. [后续加速方向](#六后续加速方向)

---

## 一、模型整体结构

Qwen3-0.6B 是一个**Decoder-only Transformer**，结构与 LLaMA 系列高度一致：

```
输入 token ids
       │
  ┌────▼────────────────────────────────────────┐
  │  Token Embedding (VocabParallelEmbedding)    │
  └────┬────────────────────────────────────────┘
       │  hidden_states: (seq_len, 1024)
       │
  ┌────▼────────────────────────────────────────┐  ×28
  │  Decoder Layer                               │
  │  ┌─────────────────────────────────────────┐│
  │  │ RMSNorm (input_layernorm)               ││
  │  │ Qwen3Attention (GQA + RoPE + QK Norm)   ││
  │  │ RMSNorm (post_attention_layernorm)       ││
  │  │ Qwen3MLP (SwiGLU)                       ││
  │  └─────────────────────────────────────────┘│
  └────┬────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────┐
  │  Final RMSNorm                               │
  └────┬────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────┐
  │  LM Head (ParallelLMHead)                    │
  └────┬────────────────────────────────────────┘
       │  logits: (batch, 151936)
  ┌────▼────────────────────────────────────────┐
  │  Sampler (Gumbel-max trick)                  │
  └─────────────────────────────────────────────┘
       │  next token ids
```

Decode layer的第一层结构如下所示（共28层）：
```python
Qwen3DecoderLayer(
  (self_attn): Qwen3Attention(
    (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
    (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
    (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
    (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
    (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
    (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
  )
  (mlp): Qwen3MLP(
    (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
    (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
    (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
    (act_fn): SiLUActivation()
  )
  (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
  (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
)
```

**代码入口**：[`nanovllm/models/qwen3.py`](../nanovllm/models/qwen3.py) — `Qwen3ForCausalLM`

---

## 二、Qwen3-0.6B 超参总览

| 超参 | 值 | 含义 |
|------|----|------|
| `hidden_size` | **1024** | 每个 token 的特征向量维度 |
| `num_attention_heads` | **16** | Q 的注意力头数 |
| `num_key_value_heads` | **8** | KV 头数（GQA，比 Q 少一半） |
| `head_dim` | **128** | 每个头的维度（= 1024×2/16，头显式指定） |
| `intermediate_size` | **3072** | MLP 中间层维度 |
| `num_hidden_layers` | **28** | Decoder 层数 |
| `vocab_size` | **151936** | 词表大小 |
| `rms_norm_eps` | 1e-6 | RMSNorm 防除零的 ε |
| `rope_theta` | 1,000,000 | RoPE 基频（越大支持越长上下文） |
| `max_position_embeddings` | 40,960 | 最大序列长度 |
| `attention_bias` | **false** | 无 QKV bias → 启用 Per-head QK Norm |
| `tie_word_embeddings` | **true** | LM Head 与 Embedding 共享权重 |
| `torch_dtype` | bfloat16 | 推理精度 |

> **总参数量估算**  
> Embedding: 151936×1024 ≈ 0.16B  
> 每层: Attn(QKV+O) + MLP(gate+up+down) ≈ (4096+1024 + 6144+3072)×1024 / 1M ≈ 14.7M × 28 ≈ 0.41B  
> 合计 ≈ **0.6B** ✓ （感兴趣的可以参考 UCSD cs234 课程）

> **Qwen3 dense模型特点**
> * GQA，KV为8个heads，Q为16个heads，减少KV cache存储
> * 采用pre-norm（梯度不容易消失），采用RMSNorm（计算更快，也更稳定）
> * MLP层的activation采用silu（门控）

---

## 三、逐层详解

### 3.1 Token Embedding — `VocabParallelEmbedding`

**代码**：[`nanovllm/layers/embed_head.py`](../nanovllm/layers/embed_head.py)

#### 数学原理

将离散 token id 转换为连续向量：

$$h_0 = W_E[x_i], \quad W_E \in \mathbb{R}^{V \times d}$$

其中 $V=151936$，$d=1024$，$x_i$ 是第 $i$ 个 token 的 id。

#### NanoVLLM 实现

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        # 按词表维度切分，每个 rank 只存一段
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

    def forward(self, x):
        # 只查本 rank 负责的 token，其他位置输出置零，再 all_reduce 汇聚
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        x = mask * (x - self.vocab_start_idx)   # 越界 id 清零防崩溃
        y = F.embedding(x, self.weight)
        y = mask.unsqueeze(1) * y               # 非本 rank 负责的位置清零
        dist.all_reduce(y)                       # SUM → 等价于"选择"
        return y
```

**关键点**：

- TP 下词表按 rank 均分：`(151936/tp, 1024)` per rank
- `mask * (x - vocab_start)` 是一个巧妙的 **越界保护**：越界 id 会变成 0，查到的是合法位置（weight[0]），但结果会被 `mask` 清零，因此不影响正确性，也不触发 index-out-of-range
- `tie_word_embeddings=true`：LM Head 直接引用 `embed_tokens.weight.data`，免去一份显存占用

---

### 3.2 RMSNorm

**代码**：[`nanovllm/layers/layernorm.py`](../nanovllm/layers/layernorm.py)

#### 数学原理

相比 LayerNorm，RMSNorm 去掉了减均值的步骤，只做 Root Mean Square 归一化：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

其中 $\gamma \in \mathbb{R}^d$ 是可学习的缩放参数（`weight`），$\epsilon$ 防止除以零。

**为什么 RMSNorm 比 LayerNorm 快？** 少了一次均值计算，且无 $\beta$ 参数。

#### NanoVLLM 实现

```python
class RMSNorm(nn.Module):
    @torch.compile  # ← 自动编译为融合 kernel
    def rms_forward(self, x):
        orig_dtype = x.dtype
        x = x.float()                             # 防止 bf16 精度不足
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))        # in-place，省显存
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile  # ← 融合了 residual add + norm，减少一次显存读写
    def add_rms_forward(self, x, residual):
        x = x.float().add_(residual.float())       # x = x + residual
        residual = x.to(orig_dtype)                # 保存残差
        # ... 再做 RMSNorm
        return x, residual
```

**关键点**：

- **Fused Residual Add + Norm**：`Qwen3DecoderLayer.forward()` 直接调用 `layernorm(hidden, residual)` 形式，把 `hidden += residual` 和 normalize 合并成一次 kernel，避免一次单独的 add kernel
- `@torch.compile` 让 PyTorch 自动做算子融合，生成更快的 kernel
- 中间转 `float()` 再乘权重，最后转回 `orig_dtype`，保证数值稳定性

---

### 3.3 Grouped-Query Attention (GQA)

**代码**：[`nanovllm/models/qwen3.py`](../nanovllm/models/qwen3.py) — `Qwen3Attention`

#### 数学原理

**标准多头注意力（MHA）**：

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**GQA（分组查询注意力）**：Q 有 $H_Q = 16$ 个头，KV 只有 $H_{KV} = 8$ 个头，每 $H_Q / H_{KV} = 2$ 个 Q 头共享一对 KV 头：

```
Q heads:    [Q0  Q1] [Q2  Q3] [Q4  Q5] [Q6  Q7] [Q8  Q9] [Q10 Q11] [Q12 Q13] [Q14 Q15]
KV heads:    [K0  V0] [K1  V1] [K2  V2] [K3  V3] [K4   V4] [K5   V5]  [K6   V6]  [K7   V7]
              ↑每组2个Q头共享同一对KV
```

**为什么用 GQA？**
MHA 的 KV Cache 大小正比于 $H_Q$；GQA 中 KV 头数更少（这里减半），**KV Cache 内存节省 50%**，decode 速度更快（KV 加载是 decode 阶段的带宽瓶颈）。

#### NanoVLLM 实现

```python
class Qwen3Attention(nn.Module):
    def __init__(self, ...):
        tp_size = dist.get_world_size()
        self.num_heads    = 16 // tp_size   # TP 后每 rank 本地 Q 头数
        self.num_kv_heads =  8 // tp_size   # TP 后每 rank 本地 KV 头数

        self.qkv_proj = QKVParallelLinear(   # 合并 Q/K/V 投影
            hidden_size=1024,
            head_size=128,
            total_num_heads=16,
            total_num_kv_heads=8,
        )
        self.o_proj = RowParallelLinear(     # 输出投影，含 all_reduce
            input_size=16*128,
            output_size=1024,
        )

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)           # (seq, local_q+local_k+local_v)
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(-1, self.num_heads,    self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)   # Per-head QK Norm（Qwen3 独有）
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)                       # Flash Attention
        output = self.o_proj(o.flatten(1, -1))       # RowParallel + all_reduce
        return output
```

**完整权重形状（TP=1）**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `qkv_proj.weight` | `(4096, 1024)` | = (16+8+8)×128 行，1024 列 |
| `o_proj.weight` | `(1024, 2048)` | 1024 行，16×128=2048 列 |

---

### 3.4 Per-Head QK Norm

**代码**：`Qwen3Attention.__init__` 中 `self.q_norm` / `self.k_norm`

Qwen3 的特有设计（当 `attention_bias=false` 时启用，0.6B 满足此条件）：

$$q_i = \text{RMSNorm}(q_i), \quad k_i = \text{RMSNorm}(k_i) \quad \forall \text{ head } i$$

每个头的 $q/k$ 在做注意力之前独立归一化。

**为什么这样做？**
- 稳定训练时 Q/K 的尺度，防止注意力分数过大导致 softmax 饱和
- 与 `attention_bias=false` 配合，QK 无偏置，Norm 代替了偏置的"中心化"作用
- `q_norm/k_norm` 的权重形状是 `(128,)`（head_dim），**在所有 rank 上完整复制**，无 TP 切分

---

### 3.5 RoPE 旋转位置编码

**代码**：[`nanovllm/layers/rotary_embedding.py`](../nanovllm/layers/rotary_embedding.py)

#### 数学原理

RoPE 通过旋转矩阵将**绝对位置信息**编码进 Q/K，使得注意力分数 $q_m \cdot k_n$ 仅依赖于**相对位置** $m-n$：

$$q_m = R_m q, \quad k_n = R_n k$$
$$q_m \cdot k_n = q^T R_m^T R_n k = q^T R_{m-n} k$$

其中旋转矩阵 $R_\theta$ 对向量的每对相邻维度施加角度为 $\theta_i$ 的旋转：

$$\theta_i = \frac{1}{10000^{2i/d}}, \quad i = 0,1,...,d/2-1$$

Qwen3-0.6B 使用 `rope_theta=1,000,000`（比原始 RoPE 的 10000 大 100 倍），让高频分量旋转更慢，**支持更长的上下文**（最大 40960 token）。

#### NanoVLLM 实现

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        # 预计算所有位置的 cos/sin 缓存：(max_pos, 1, head_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        freqs = torch.einsum("i,j->ij", positions, inv_freq)     # (max_pos, head_dim/2)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # (max_pos, head_dim)
        self.register_buffer("cos_sin_cache", cos_sin_cache.unsqueeze(1))

    @torch.compile   # ← 融合 index + rotate 操作
    def forward(self, positions, query, key):
        cos_sin = self.cos_sin_cache[positions]    # 按当前 token 位置取缓存
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key   = apply_rotary_emb(key,   cos, sin)
        return query, key
```

**关键点**：

- 用 `register_buffer` 预存所有位置的 $\cos/\sin$ 表，推理时只做 index 查找，避免重复计算
- `positions` 在 prefill 时是 `[0,1,2,...,seq_len-1]`，在 decode 时是当前序列长度 `[len(seq)-1]`
- `@lru_cache(1)` 保证同一配置复用同一个 RoPE 实例，避免重复构造大 buffer

---

### 3.6 Paged KV Cache + Flash Attention

**代码**：[`nanovllm/layers/attention.py`](../nanovllm/layers/attention.py)

#### KV Cache 的必要性

在 decode 阶段，每生成一个新 token，都需要对所有历史 token 做注意力。若不缓存，则每步都要重新计算所有历史 K/V，代价是 $O(T^2)$。KV Cache 把历史 K/V 存起来，每步只做 $O(T)$ 的计算。

#### Paged KV Cache

借鉴操作系统分页内存的思想：

```
物理 KV Cache（连续大块）：
┌────────────┬────────────┬────────────┬────────────┐
│  Block 0   │  Block 1   │  Block 2   │  Block 3   │  ...
│ 256 tokens │ 256 tokens │ 256 tokens │ 256 tokens │
└────────────┴────────────┴────────────┴────────────┘

序列的逻辑 KV（可以不连续）：
Seq A: 用了 Block 0（token 0-255） + Block 2（token 256-511）
Seq B: 用了 Block 1（token 0-255）
```

`slot_mapping[i]` 告诉 Triton kernel：第 $i$ 个 token 的 KV 应写到物理 Cache 的哪个 slot。

```
slot = block_table[seq_id][block_id] * block_size + block_offset
```

**Triton KV Cache 写入 kernel**：

```python
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)              # 每个 token 启动一个 program
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return               # prefill prefix 缓存命中，跳过
    key   = tl.load(key_ptr   + idx * key_stride   + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)
```

`D = num_kv_heads × head_dim`（把头和 head_dim 压成一维，方便地址计算）。

#### Flash Attention 两种调用形式

```python
def forward(self, q, k, v):
    # 1. 先将当前 step 的 K/V 写入 KV Cache
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    if context.is_prefill:
        # Prefill：变长序列，用 varlen 版本（不同请求的序列拼在一起）
        o = flash_attn_varlen_func(q, k, v,
            cu_seqlens_q=context.cu_seqlens_q,   # 累积序列长度，定位每条序列边界
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            softmax_scale=self.scale, causal=True)
    else:
        # Decode：每条序列只有 1 个新 token，从 KV Cache 取历史
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens,   # 每条序列当前已有多少 token
            block_table=context.block_tables,     # 物理块表
            softmax_scale=self.scale, causal=True)
    return o
```

**KV Cache 形状（TP=1）**：

```
kv_cache: (2, 28, num_blocks, block_size=256, num_kv_heads=8, head_dim=128)
 │           │    │              │             │               └─ 每头特征维度
 │           │    │              │             └─ KV头数（TP后 8//tp）
 │           │    │              └─ 每个 block 容纳的 token 数
 │           │    └─ 物理 block 总数（由剩余显存决定）
 │           └─ 28 层
 └─ 0=K, 1=V
```

---

### 3.7 SwiGLU MLP

**代码**：[`nanovllm/models/qwen3.py`](../nanovllm/models/qwen3.py) — `Qwen3MLP`，[`nanovllm/layers/activation.py`](../nanovllm/layers/activation.py)

#### 数学原理

标准 FFN：$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1) W_2 + b_2$

SwiGLU（Qwen3 使用）：

$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\big) W_{\text{down}}$$

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1+e^{-z}}$$

相比 ReLU，SiLU 是平滑的，且无梯度消失问题；门控 $W_{\text{gate}}$ 决定"开放哪些神经元"，$W_{\text{up}}$ 提供幅度，两者逐元素相乘形成**选择性激活**。

#### NanoVLLM 实现

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=3072):
        # gate_proj + up_proj 合并为一个矩阵，一次 GEMM 完成两个投影
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size]
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)   # (seq, 6144) = gate(3072) + up(3072)
        x = self.act_fn(gate_up)         # chunk + silu + mul → (seq, 3072)
        x = self.down_proj(x)            # RowParallel + all_reduce → (seq, 1024)
        return x
```

```python
class SiluAndMul(nn.Module):
    @torch.compile   # ← 融合 chunk + silu + mul 三个操作
    def forward(self, x):
        x, y = x.chunk(2, -1)   # 拆分 gate 和 up
        return F.silu(x) * y
```

**关键点**：

- `gate_proj` 和 `up_proj` **合并成一个大矩阵** `gate_up_proj`，一次 cuBLAS GEMM 完成两个投影，比两次独立 GEMM 更高效（减少 kernel launch 开销）
- `SiluAndMul` 用 `@torch.compile` 融合，避免中间 tensor 的显存分配

---

### 3.8 LM Head + 采样

**代码**：[`nanovllm/layers/embed_head.py`](../nanovllm/layers/embed_head.py) — `ParallelLMHead`，[`nanovllm/layers/sampler.py`](../nanovllm/layers/sampler.py)

#### LM Head

将 hidden state 映射回词表空间：

$$\text{logits} = h \cdot W_E^T, \quad W_E \in \mathbb{R}^{V \times d}$$

注意：`tie_word_embeddings=true` 使 LM Head 重用 Embedding 权重（$W_E$ 转置），节约 ~0.15B 参数的显存。

**Prefill 阶段优化**：只计算每条序列**最后一个 token** 的 logits：

```python
def forward(self, x):
    if context.is_prefill:
        last_indices = context.cu_seqlens_q[1:] - 1  # 取每条序列最后一个 token
        x = x[last_indices].contiguous()             # (batch_size, 1024)，而非 (seq_len, 1024)
    logits = F.linear(x, self.weight)   # (batch, vocab/tp)
    dist.gather(logits, ..., dst=0)     # 只 rank 0 收集完整 logits
```

**为什么 prefill 只需最后一个 token？** 模型生成是自回归的，prefill 结束后，我们只需要预测序列中下一个（即第 seq_len+1 个）token，而这仅由最后位置的 hidden state 决定。

#### Sampler

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(1))  # 温度缩放
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max trick：等价于 multinomial sampling，但可以用 argmax 实现，更快
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

**Gumbel-max trick**：从分类分布采样，一般需要 `torch.multinomial`（串行）；Gumbel-max 等价于对每个类加独立 Gumbel 噪声后取 argmax（并行），公式为：

$$\hat{x} = \arg\max_i \left[\log p_i - \log(-\log u_i)\right], \quad u_i \sim \text{Uniform}(0,1)$$

代码中 `exponential_(1)` 生成的是 $-\log u_i$，整体等价于上式，可以批量并行完成。

---

## 四、关键实现细节与注意事项

### 4.1 `@torch.compile` 的使用策略

NanoVLLM 对**小而高频的算子**使用 `@torch.compile`，而不是整个模型：

| 函数 | 原因 |
|------|------|
| `RMSNorm.rms_forward` | 多个 elementwise 操作，compile 后融合成单 kernel |
| `RMSNorm.add_rms_forward` | add + norm 融合，减少一次显存读写 |
| `SiluAndMul.forward` | chunk + silu + mul 融合 |
| `RotaryEmbedding.forward` | index + rotate 融合 |
| `Sampler.forward` | temperature_scale + softmax + sampling 融合 |

> ⚠️ **注意**：`@torch.compile` 第一次调用有 JIT 编译开销（通常 10-60 秒），这是 `warmup_model()` 的作用之一。

### 4.2 CUDA Graph 机制

`model_runner.capture_cudagraph()` 为不同 batch size 录制 CUDA Graph：

```python
graph_bs = [1, 2, 4, 8, 16, 32, ..., 512]
for bs in reversed(graph_bs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool):
        outputs[:bs] = model(input_ids[:bs], positions[:bs])
    graphs[bs] = graph
```

推理时找到 `>= bs` 的最小 graph 执行 `graph.replay()`，延迟接近**单次 kernel launch**。

> ⚠️ **限制**：CUDA Graph 只支持 decode 阶段（batch size 固定）；prefill 阶段 seq_len 可变，只能 eager 执行。因此推理时判断：
> ```python
> if is_prefill or enforce_eager or bs > 512:
>     return model(...)          # eager
> else:
>     graph.replay(); return ... # CUDA Graph
> ```

### 4.3 weight_loader 的设计哲学

每个 `nn.Parameter` 上挂载了专属的 `weight_loader` 函数：

```python
# LinearBase.__init__()
self.weight.weight_loader = self.weight_loader
```

`load_model()` 只需遍历文件，不感知 TP 切分逻辑：

```python
weight_loader = getattr(param, "weight_loader", default_weight_loader)
weight_loader(param, loaded_weight[, shard_id])
```

这是一种**策略模式（Strategy Pattern）**的应用：切分策略封装在各层内部，加载层完全解耦。

### 4.4 Prefill vs Decode 的统一 Context

`Context` 是一个全局单例，在每次 forward 前通过 `set_context()` 更新，在 Attention 层中通过 `get_context()` 读取：

```python
@dataclass
class Context:
    is_prefill: bool
    cu_seqlens_q: Tensor   # Prefill 专用：累积序列长度
    slot_mapping: Tensor   # 两阶段都用：KV 写入位置
    context_lens: Tensor   # Decode 专用：每序列历史长度
    block_tables: Tensor   # Decode 专用：物理块表
```

这避免了在 forward 函数签名中传递大量 attention metadata，保持了模型代码的简洁。

### 4.5 常见注意事项

| 注意事项 | 说明 |
|---------|------|
| **TP 整除约束** | `num_attention_heads` 和 `num_key_value_heads` 必须能被 `tp_size` 整除；vocab_size 也需整除 |
| **BF16 精度** | RMSNorm 内部转 float32 再转回，避免 bf16 精度不足（bf16 只有约 3 位有效小数） |
| **bias 只加一次** | `RowParallelLinear` 中 bias 只在 `rank 0` 加，`all_reduce` 后不会被放大；Qwen3-0.6B 的 o_proj/MLP 均无 bias，此处不影响 |
| **KV Cache block_size** | 必须是 256 的倍数（`config.py` 断言），与 flash_attn 内部 page attention 的对齐要求有关 |
| **CUDA Graph 与 KV Cache** | `graph_vars["block_tables"]` 必须足够大（`max_num_blocks`），replay 时原地替换数据 |
| **tie_word_embeddings** | `lm_head.weight.data = embed_tokens.weight.data`（共享 data 指针），修改其中一个会影响另一个 |

---

## 五、后续加速方向

### 5.1 算子融合（已有 / 可扩展）

#### 已实现
- `RMSNorm + residual add`：`add_rms_forward` 用 `@torch.compile` 融合，减少 1 次 elementwise kernel
- `SiluAndMul`：`chunk + silu + mul` 融合
- `gate_proj + up_proj`：合并为一个 GEMM（`MergedColumnParallelLinear`）

#### 可进一步融合的方向

**（1）QKV 融合 + RoPE 融合**

目前流程：`qkv_proj` → `split` → `q_norm` / `k_norm` → `rotary_emb`（4 步）

可以写一个 Triton kernel，将 `Q/K RMSNorm + RoPE` 融合成单 kernel：

```python
@triton.jit
def fused_qk_norm_rope_kernel(q_ptr, k_ptr, norm_w_ptr, cos_ptr, sin_ptr, ...):
    # 1. 读取 q/k
    # 2. 做 RMSNorm（逐 head）
    # 3. 施加 RoPE 旋转
    # 4. 写回
```

收益：减少 3 次 kernel launch + 3 次 HBM 读写。

**（2）Fused LayerNorm + Linear**

将 `post_attn_layernorm` 和 `gate_up_proj` 的 GEMM 融合——先 norm 输出到 register，直接乘进 GEMM 的 A 矩阵。此技术在 FlashAttention3 中被称为 **epilogue fusion**。

**（3）Sampler 融合**

目前 `temperature_scale → softmax → sampling` 是 3 步；可以写成单 pass kernel，配合 `top-p` 过滤等。

---

### 5.2 Triton 自定义 Kernel 方向

#### （1）Fused RMSNorm Triton Kernel

替换当前 `@torch.compile` 实现，精细控制 SRAM 使用：

```python
@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N,          # M=batch, N=hidden_size
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * N + tl.arange(0, BLOCK_N))
    var = tl.sum(x * x, axis=0) / N
    x_norm = x * tl.rsqrt(var + 1e-6)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_N))
    tl.store(out_ptr + row * N + tl.arange(0, BLOCK_N), x_norm * w)
```

#### （2）扩展 store_kvcache 支持 FP8

当前 NanoVLLM 的 KV Cache 是 BF16。FP8 KV Cache 可以再减半显存，存储更多历史上下文：

```python
@triton.jit
def store_kvcache_fp8_kernel(...):
    key = tl.load(...)
    # 动态量化到 FP8
    scale = tl.max(tl.abs(key)) / 448.0   # FP8 E4M3 最大值
    key_fp8 = (key / scale).to(tl.float8e4m3)
    tl.store(k_cache_ptr + ..., key_fp8)
    tl.store(k_scale_ptr + ..., scale)
```

#### （3）Decode 阶段 Fused Attention + KV Cache 读取

`flash_attn_with_kvcache` 已经是融合 kernel，但可以进一步定制以支持：
- **Speculative Decoding**：同时验证多个 draft tokens
- **RadixAttention / Prefix Caching**：在 kernel 内处理 prefix 命中逻辑

---

### 5.3 CUDA 级别优化方向

#### （1）Persistent Kernel + Warp Specialization

将 Decode 阶段的 `Attention + o_proj + Norm + MLP` 组织成一个**持久化 kernel**，不同 warp 负责不同操作，消除 kernel launch 延迟（参考 DeepSpeed-FastGen）。

#### （2）W4A16/W8A16 量化

对权重做 INT4/INT8 量化，推理时反量化后做 GEMM：
- `gate_up_proj.weight`：BF16(3072×1024×2B) = 6MB → INT4(1.5MB)，KV 带宽瓶颈转移为 GEMM 计算
- 可用 bitsandbytes / GPTQ / AWQ 方案替换 `weight_loader`，在加载时直接存储量化权重

#### （3）All-Reduce 优化：Ring All-Reduce vs. Tree All-Reduce

当前 `dist.all_reduce` 使用 NCCL 默认策略（NVLink 机器上是 Ring）。对于小 tensor（hidden_size=1024），可以评估：
- **Reduce-Scatter + All-Gather**：Megatron-LM 的做法，适合大 batch
- **自定义 NCCL communicator**：绑定 NVLink 拓扑，减少跳数

#### （4）投机解码（Speculative Decoding）

在当前框架上集成 Draft 模型（如 Qwen3-0.5B 作为草稿，0.6B 作为验证），利用 `flash_attn_with_kvcache` 的批量验证特性，在不降低质量的前提下提升 3-5× 吞吐。

---

### 5.4 系统级优化方向

| 方向 | 描述 | 预期收益 |
|------|------|---------|
| **Chunked Prefill** | 将长 prefill 拆成小 chunk，与 decode 混合调度，避免 decode 饥饿 | 延迟 P99 改善 |
| **Prefix/RadixTree KV Cache** | 复用相同 prompt prefix 的 KV，SGLang 已实现 | 多轮对话吞吐 2-4× |
| **Continuous Batching 优化** | 当前调度器已支持，可细化 bucket allocation | 显存利用率提升 |
| **FP8 训练 + 推理** | H100 支持原生 FP8 GEMM，end-to-end FP8 可达 BF16 速度 2× | 需要精度验证 |
| **Disaggregated Prefill** | Prefill 和 Decode 用不同机器，解决 compute bound vs memory bound 的矛盾 | 高并发场景 |

---

## 参考资料

1. **Qwen3 论文**：[arxiv 2505.09388](https://arxiv.org/abs/2505.09388)
2. **RoPE 原文**：RoFormer: Enhanced Transformer with Rotary Position Embedding
3. **GQA 原文**：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
4. **Flash Attention 2/3**：Dao et al., 2022/2023/2024
5. **Megatron-LM TP**：Shoeybi et al., Efficient Large-Scale Language Model Training
6. **vLLM Paged Attention**：Kwon et al., 2023
7. **NanoVLLM 解读**：[知乎](https://zhuanlan.zhihu.com/p/1977336847567983629)
8. **TP 并行详解**：见本仓库 [`docs/tp_demos/README.md`](tp_demos/README.md)
