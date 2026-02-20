# NanoVLLM Tensor Parallelism 完整技术梳理

> 文档由Claude Sonnet4.6生成

## 总览：三层架构

```
┌──────────────────────────────────────────────────────────────┐
│                    进程管理层（控制面）                         │
│  LLMEngine → mp.spawn → ModelRunner(rank=0..N-1)             │
│  SharedMemory + multiprocessing.Event  ←→  rank 0 广播指令    │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│                    NCCL 通信层（数据面）                        │
│  all_reduce (RowParallel/Embedding)                           │
│  gather     (LMHead logits)                                   │
│  barrier    (初始化同步)                                       │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│                    模型权重切分层                               │
│  ColumnParallel / RowParallel / QKVParallel                   │
│  VocabParallelEmbedding / ParallelLMHead                      │
│  KV Cache 按 tp_size 切分                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 一、进程管理与共享内存（demo2_shared_memory.py）

### 1.1 进程启动流程

```python
# llm_engine.py  LLMEngine.__init__()
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.events.append(event)
self.model_runner = ModelRunner(config, 0, self.events)  # rank 0 在主进程
```

关键点：
- `spawn` 模式：子进程从头 fork，不继承父进程的 CUDA 上下文，是 NCCL 的要求
- rank 0 在**主进程**中运行，rank 1..N-1 在子进程中运行
- `LLMEngine` 的 `step()` → `model_runner.call()` 触发 rank 0 的执行
- rank1-N的执行，是在`ModelRunner`类的`__init__()`方法中，通过调用`loop()` 方法循环等待rank0传来的method name和args。

### 1.2 SharedMemory 通信协议

```
shm 内存布局（最大 2^20 = 1 MB）：
┌────────────┬──────────────────────────────────┐
│  [0:4]     │  payload 长度 N（4 字节小端整数）   │
│  [4:4+N]   │  pickle.dumps([method_name, *args])│
└────────────┴──────────────────────────────────┘
```

**rank 0 写（write_shm）→ event.set() 通知各 rank**
**rank 1..N 在 event.wait() 阻塞 → 读取 → event.clear()**

下面两段代码分别是rank0的write_shm和rank1-N的read_shm。

```python
def call(self, method_name, *args):
        # TODO(leon): 多GPU推理，后续dump
        # Tensor parallelism下，rank 0进程通过共享内存与其他rank通信
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)
```

```python
# model_runner.py  ModelRunner.loop()  (rank 1..N 的工作循环)
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)     # 与 rank 0 同步执行
        if method_name == "exit":
            break
```

**为什么不用 NCCL 传方法名？**
- NCCL 是 GPU 通信库，只处理张量数据
- 方法名是 Python 字符串，用 pickle + SharedMemory 更轻量
- Event 是多进程同步原语，比 NCCL barrier 开销小得多

### 1.3 SharedMemory 生命周期

```
rank 0: create(name="nanovllm") → barrier → ... → close() → unlink()
rank N: barrier                 → open()  → ... → close()
```
- `close()` 只是解除本进程的映射引用
- `unlink()` 才是真正释放 POSIX 共享内存，只能由创建者调用一次

---

## 二、NCCL 通信（demo1_nccl_communication.py）

### 2.1 进程组初始化

```python
# model_runner.py  ModelRunner.__init__()
dist.init_process_group("nccl", "tcp://localhost:2333",
                         world_size=self.world_size, rank=rank)
torch.cuda.set_device(rank)
```

每个 rank 在自己的 CUDA 设备上初始化，所有 rank 通过 TCP rendezvous 建立 NCCL 通信域。

### 2.2 三种通信原语及其使用场景

| 原语 | 位置 | 语义 |
|------|------|------|
| `dist.all_reduce(SUM)` | `RowParallelLinear.forward()` | 各 rank 部分积求和，结果复制到每个 rank |
| `dist.all_reduce(SUM)` | `VocabParallelEmbedding.forward()` | 各 rank 非零 embedding 汇聚（每位置只有一个 rank 非零） |
| `dist.gather(dst=0)` | `ParallelLMHead.forward()` | 各 rank logit 分片汇聚到 rank 0（节省带宽，只 rank 0 做 sample） |
| `dist.barrier()` | `ModelRunner.__init__()` | SharedMemory 创建/打开之间的同步屏障 |

### 2.3 通信代价分析

每个 Transformer 层只有 **2 次 all_reduce**：
1. Attention o_proj 之后（RowParallel）
2. MLP down_proj 之后（RowParallel）

通信量 = `2 × BATCH × hidden_size × 2bytes`（一来一回，BF16）

---

## 三、模型层 TP 切分（demo3_model_layers.py）

### 3.1 Megatron-LM 风格的 Column-Row 配对

```
                     ┌──── rank 0 ────┐  ┌──── rank 1 ────┐
input (replicated)   │  W_col_0[x]    │  │  W_col_1[x]    │  ← ColumnParallel (无通信)
                     └───────┬────────┘  └────────┬───────┘
                             │ partial_out_0       │ partial_out_1
                     ┌───────▼────────────────────▼───────┐
                     │        W_row @ concat_partial        │  ← RowParallel
                     │           + all_reduce(SUM)          │
                     └──────────────────────────────────────┘
                          output (replicated, same value on all ranks)
```

### 3.2 各层切分维度

| 层 | 权重形状 | TP 切分维度 | weight_loader 逻辑 |
|----|---------|-----------|-------------------|
| `ColumnParallelLinear` | `(out, in)` | dim=0（输出） | `narrow(0, rank*shard, shard)` |
| `RowParallelLinear` | `(out, in)` | dim=1（输入） | `narrow(1, rank*shard, shard)` |
| `MergedColumnParallelLinear` | `(gate+up, in)` | dim=0，gate/up 分区段 | 分两次写入各自偏移 |
| `QKVParallelLinear` | `(q+k+v, hidden)` | 按注意力头 | 分 "q"/"k"/"v" 三次调用，写入不同偏移 |
| `VocabParallelEmbedding` | `(vocab, embed)` | dim=0（词表） | `narrow(0, rank*per, per)` |
| `ParallelLMHead` | `(vocab, hidden)` | dim=0（词表） | 同 Embedding |

### 3.3 QKV 切分细节（GQA 场景）

```
完整 QKV 权重（Qwen3 GQA 示例）：
  Q: (total_Q_heads × head_dim, hidden)
  K: (total_KV_heads × head_dim, hidden)   total_KV < total_Q
  V: (total_KV_heads × head_dim, hidden)

合并后：
  QKV_weight ∈ R^{(total_Q + 2*total_KV)*head_dim × hidden}

rank i 持有：
  Q → rows [i*nQ_per, (i+1)*nQ_per) × head_dim
  K → rows [nQ_per*tp + i*nKV_per, ...] × head_dim
  V → rows [nQ_per*tp + nKV_per*tp + i*nKV_per, ...] × head_dim
```

### 3.4 KV Cache TP 内存节省

```python
# model_runner.py  allocate_kv_cache()
num_kv_heads = hf_config.num_key_value_heads // self.world_size
kv_cache = torch.empty(
    2,                           # K 和 V
    hf_config.num_hidden_layers, # L 层
    config.num_kvcache_blocks,   # B 个 block
    self.block_size,             # T tokens/block
    num_kv_heads,                # H 每 rank 负责的头数
    head_dim                     # D
)
```

TP=2 时：KV Cache 内存减半；TP=4 时：减为 1/4。

### 3.5 weight_loader 机制（避免重复加载）

```python
# utils/loader.py  load_model()
for weight_name in safetensors_file.keys():
    param = model.get_parameter(param_name)
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_tensor, shard_id)  # 每个 rank 只写自己的分片
```

每个参数对象上挂载了专属的 `weight_loader`，在 `LinearBase.__init__()` 中注册：
```python
self.weight.weight_loader = self.weight_loader
```

这样 `load_model` 只需遍历文件，不需要知道 TP 切分逻辑，各层自己处理。

---

## 四、Qwen3-0.6B 逐层 TP 并行分析（TP=2）

> **Qwen3-0.6B 模型超参**（来自 `config.json`）
>
> | 参数 | 值 |
> |------|----|
> | `hidden_size` | 1024 |
> | `num_attention_heads` (Q) | 16 |
> | `num_key_value_heads` (KV, GQA) | 8 |
> | `head_dim` | 128 |
> | `intermediate_size` | 3072 |
> | `num_hidden_layers` | 28 |
> | `vocab_size` | 151936 |
> | `attention_bias` | false → 启用 per-head QK Norm |
> | `tie_word_embeddings` | true → lm_head 与 embed_tokens 共享权重 |

---

### 4.1 Embedding 层：`VocabParallelEmbedding`

**代码位置**：`layers/embed_head.py`，`models/qwen3.py: Qwen3Model.embed_tokens`

```
完整权重：(151936, 1024)
          ↓ 按 vocab 维度（dim=0）切分
rank 0 持有：(75968, 1024)   ← token id [0,      75967]
rank 1 持有：(75968, 1024)   ← token id [75968, 151935]
```

**forward 逻辑（TP=2）**：
```python
# 输入：token_ids (seq_len,)，两个 rank 都持有完整副本
mask = (token_ids >= vocab_start) & (token_ids < vocab_end)  # 本 rank 负责哪些 token
safe_ids = mask * (token_ids - vocab_start)   # 越界 id 置 0 防止崩溃
y = F.embedding(safe_ids, local_weight)       # (seq_len, 1024) 但越界位为随机值
y = mask.unsqueeze(1) * y                     # 越界位清零 → 只有本 rank 负责的位置非零
dist.all_reduce(y, SUM)                       # 每个位置只有一个 rank 非零，SUM 等价于"选择"
# 输出：(seq_len, 1024)，两 rank 结果相同（replicated）
```

**通信**：1 次 `all_reduce`，通信量 = `seq_len × 1024 × 2 bytes`

---

### 4.2 Decoder Layer × 28

每层结构：`input_layernorm → Attention → post_attn_layernorm → MLP`

#### 4.2.1 RMSNorm（`input_layernorm` / `post_attention_layernorm`）

**代码位置**：`layers/layernorm.py`，`models/qwen3.py: Qwen3DecoderLayer`

```
权重：(1024,) — 在所有 rank 上完整复制，无切分
前向：本地计算，无任何通信
```

NanoVLLM 的 `RMSNorm` 使用 fused 实现，支持 **in-place residual 融合**（`hidden, residual = layernorm(hidden, residual)`），避免显式的 residual add 操作。

---

#### 4.2.2 Attention：`QKVParallelLinear`（`qkv_proj`）

**代码位置**：`layers/linear.py: QKVParallelLinear`

```
HuggingFace safetensors 存储（3 个独立文件）：
  q_proj.weight : (16×128,  1024) = (2048, 1024)
  k_proj.weight : ( 8×128,  1024) = (1024, 1024)
  v_proj.weight : ( 8×128,  1024) = (1024, 1024)

packed_modules_mapping 将其合并加载进 qkv_proj：
  qkv_proj.weight 完整逻辑视图：(2048+1024+1024, 1024) = (4096, 1024)
                                  [─── Q ───][─K─][─V─]

TP=2 每 rank 实际存储：(2048, 1024)
  rank 0：Q[0:1024, :] + K[0:512, :]  + V[0:512, :]   ← 8个Q头 + 4个KV头
  rank 1：Q[1024:,  :] + K[512:, :]   + V[512:, :]    ← 8个Q头 + 4个KV头
```

**weight_loader 三次调用**（`loader.py: load_model` 通过 `packed_modules_mapping` 驱动）：
```python
weight_loader(param, q_proj_weight, "q")   # narrow(0, 0,    1024) ← rank 0 的 Q 分片
weight_loader(param, k_proj_weight, "k")   # narrow(0, 1024,  512) ← rank 0 的 K 分片
weight_loader(param, v_proj_weight, "v")   # narrow(0, 1536,  512) ← rank 0 的 V 分片
```

**forward（无通信）**：
```python
qkv = F.linear(hidden, local_weight)       # (seq_len, 2048) 本 rank 的 q+k+v
q, k, v = qkv.split([1024, 512, 512], -1)
q = q.view(-1, 8,  128)   # 8  Q  heads per rank
k = k.view(-1, 4,  128)   # 4  KV heads per rank
v = v.view(-1, 4,  128)
```

---

#### 4.2.3 Attention：`q_norm` / `k_norm`（Per-head QK Norm）

**代码位置**：`models/qwen3.py: Qwen3Attention`（仅当 `attention_bias=false` 时启用，Qwen3-0.6B 满足此条件）

```
q_norm.weight : (128,) — replicated，作用在 q 的 head_dim 维度
k_norm.weight : (128,) — replicated，作用在 k 的 head_dim 维度

q = q_norm(q)   # (seq_len, 8,  128) → 每个 Q head 独立 normalize
k = k_norm(k)   # (seq_len, 4,  128)
无通信
```

---

#### 4.2.4 Attention：RoPE 旋转位置编码

**代码位置**：`layers/rotary_embedding.py`

```
无可学习参数，本地计算，无通信
q, k = rotary_emb(positions, q, k)   # 各 rank 独立对本地切片施加 RoPE
```

---

#### 4.2.5 Attention：Flash Attention + KV Cache 写入

**代码位置**：`layers/attention.py: Attention.forward`

```
每个 rank 独立运行 Flash Attention，只处理本 rank 的注意力头：
  输入：q (seq_len, 8, 128)，k (seq_len, 4, 128)，v (seq_len, 4, 128)

KV Cache 分配（model_runner.py: allocate_kv_cache）：
  kv_cache shape：(2, 28, num_blocks, block_size, 4, 128)
                                                  ↑
                                             每 rank 只有 4 个 KV 头（= 8 // 2）
  TP=2 时每 rank KV Cache 内存为单卡的 1/2

Triton kernel (store_kvcache_kernel) 将当前 step 的 k,v 写入对应 slot：
  slot = block_table[seq_id][block_id] * block_size + offset

Flash Attention 调用：
  prefill → flash_attn_varlen_func(q, k, v, ...)         # 无 KV cache 读取
  decode  → flash_attn_with_kvcache(q, k_cache, v_cache) # 从 KV cache 读历史

输出：o (seq_len, 8, 128) → flatten → (seq_len, 1024)
无进程间通信
```

---

#### 4.2.6 Attention：`RowParallelLinear`（`o_proj`）

**代码位置**：`layers/linear.py: RowParallelLinear`

```
完整权重：(1024, 2048)   行=output=1024，列=input=2048
          ↓ 按输入维度（dim=1）切分
rank 0 持有：(1024, 1024)  ← 对应 8 个 Q head 的输出
rank 1 持有：(1024, 1024)  ← 对应另外 8 个 Q head 的输出
```

**forward（含 1 次 all_reduce）**：
```python
# 每个 rank 的输入恰好是本 rank attention 的输出 (seq_len, 1024)
partial_y = F.linear(attn_output_local, local_weight)   # (seq_len, 1024) 部分积
# bias 只在 rank 0 加（o_proj 无 bias，此处无影响）
dist.all_reduce(partial_y, SUM)
# 输出：(seq_len, 1024) replicated ← 全部 rank 结果相同
```

**这是整个 Attention 子块中唯一的通信点。**

---

#### 4.2.7 MLP：`MergedColumnParallelLinear`（`gate_up_proj`）

**代码位置**：`layers/linear.py: MergedColumnParallelLinear`

```
HuggingFace 存储（2 个独立文件）：
  gate_proj.weight : (3072, 1024)
  up_proj.weight   : (3072, 1024)

packed_modules_mapping 合并加载进 gate_up_proj：
  gate_up_proj.weight 完整逻辑视图：(6144, 1024)
                                     [── gate ──][── up ──]

TP=2 每 rank 实际存储：(3072, 1024)
  rank 0：gate[0:1536, :] + up[0:1536, :]   ← 各取前一半
  rank 1：gate[1536:,  :] + up[1536:, :]    ← 各取后一半
```

**weight_loader 两次调用**：
```python
weight_loader(param, gate_proj_weight, 0)   # shard_id=0：gate 的本 rank 分片
weight_loader(param, up_proj_weight,   1)   # shard_id=1：up  的本 rank 分片
```

**forward（无通信）**：
```python
gate_up = F.linear(hidden, local_weight)    # (seq_len, 3072) 本 rank gate+up
gate, up = gate_up.chunk(2, dim=-1)         # 各 (seq_len, 1536)
x = F.silu(gate) * up                      # SiluAndMul，(seq_len, 1536)
```

---

#### 4.2.8 MLP：`RowParallelLinear`（`down_proj`）

**代码位置**：`layers/linear.py: RowParallelLinear`

```
完整权重：(1024, 3072)
          ↓ 按输入维度（dim=1）切分
rank 0 持有：(1024, 1536)
rank 1 持有：(1024, 1536)
```

**forward（含 1 次 all_reduce）**：
```python
partial_y = F.linear(activated_local, local_weight)   # (seq_len, 1024)
dist.all_reduce(partial_y, SUM)
# 输出：(seq_len, 1024) replicated
```

**这是整个 MLP 子块中唯一的通信点。**

---

### 4.3 最终 RMSNorm

```
权重：(1024,) replicated，本地计算，无通信
```

---

### 4.4 LM Head：`ParallelLMHead`

**代码位置**：`layers/embed_head.py: ParallelLMHead`

```
完整权重：(151936, 1024) — 与 embed_tokens 共享（tie_word_embeddings=true）
TP=2 每 rank：(75968, 1024)
```

**Prefill 阶段特殊处理**：
```python
# 只对每个序列的最后一个 token 计算 logits（其余位置的 logits 无用）
last_indices = cu_seqlens_q[1:] - 1
x = hidden_states[last_indices]   # (batch, 1024)
```

**forward（含 1 次 gather）**：
```python
partial_logits = F.linear(x, local_weight)   # (batch, 75968) 本 rank 负责的词表片段
if rank == 0:
    all_logits = [torch.empty_like(partial_logits)] * tp_size
dist.gather(partial_logits, all_logits, dst=0)
if rank == 0:
    logits = torch.cat(all_logits, dim=-1)   # (batch, 151936) 完整词表 logits
    # 后续 Sampler 只在 rank 0 运行
```

**为何用 gather 而非 all_reduce？**  
只有 rank 0 需要做采样，gather 的通信量是 all_gather 的一半。

---

### 4.5 TP=2 通信总览（每次 forward）

| 位置 | 原语 | 通信量（BF16，batch=B，seq=S） |
|------|------|-------------------------------|
| Embedding | `all_reduce` | `S × 1024 × 2B = 2KB×S` |
| Attn o_proj × 28 | `all_reduce` | `S × 1024 × 2B × 28` |
| MLP down_proj × 28 | `all_reduce` | `S × 1024 × 2B × 28` |
| LM Head | `gather` | `B × 75968 × 2B`（仅 decode 阶段 B 很小）|

**总 all_reduce 次数 = 1（embedding）+ 28（attn）+ 28（mlp）= 57 次，但均为小张量，延迟可被 GPU 计算掩盖。**

---

## 五、TP 并行完整流程图（伪代码）

### 5.1 系统级：进程启动与控制流

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LLMEngine.__init__()                                                   │
│                                                                         │
│   ctx = mp.get_context("spawn")                                         │
│   for rank in [1, ..., tp_size-1]:                                      │
│       event = ctx.Event()                                               │
│       process = ctx.Process(                                            │
│           target=ModelRunner,   ← 子进程入口                            │
│           args=(config, rank, event)                                    │
│       )                                                                 │
│       process.start()  ─────────────────────────────────────────────┐  │
│       events.append(event)                                          │  │
│                                                                     │  │
│   model_runner = ModelRunner(config, rank=0, events)  ← 主进程      │  │
└─────────────────────────────────────────────────────────────────────│──┘
                                                                      │
          子进程: ModelRunner(config, rank=i, event)  ◄───────────────┘
```

---

### 5.2 ModelRunner 初始化流程（每个 rank 独立执行）

```
ModelRunner.__init__(config, rank, event)
│
├── dist.init_process_group("nccl", "tcp://localhost:2333",
│       world_size=tp_size, rank=rank)
├── torch.cuda.set_device(rank)
│
├── Qwen3ForCausalLM(hf_config)          ← 在 GPU 上构造模型骨架（空权重）
│   每层的线性层已通过 dist.get_rank/world_size 确定本 rank 的 weight 形状
│
├── load_model(model, config.model)      ← 从 safetensors 加载权重
│   for each param in safetensors:
│       param.weight_loader(param, full_weight[, shard_id])
│           ← 每个 rank 只写入自己的分片，无通信
│
├── warmup_model()                       ← 空跑一次，测量峰值显存
├── allocate_kv_cache()                  ← 按剩余显存分配 KV cache
│   kv_cache: (2, 28, blocks, block_sz, num_kv_heads//tp, 128)
│
├── capture_cudagraph() [若非 eager]     ← 录制不同 batch size 的 CUDA Graph
│
└── [rank == 0]:                         ← 控制面初始化
│       shm = SharedMemory(create=True, size=2^20)
│       dist.barrier()
│       进入正常推理流程
    [rank >= 1]:
        dist.barrier()
        shm = SharedMemory(name="nanovllm")  ← 打开 rank 0 创建的 shm
        self.loop()                          ← 进入等待循环（阻塞）
```

---

### 5.3 推理调用流（rank 0 视角）

```
LLMEngine.step()
│
└── model_runner.call("run", seqs, is_prefill)
    │
    ├── [rank 0] write_shm("run", seqs, is_prefill)
    │       shm.buf = [4字节长度 | pickle([method_name, seqs, is_prefill])]
    │       for each event: event.set()     ← 唤醒 rank 1..N
    │
    ├── [rank 0] 同步执行 self.run(seqs, is_prefill)
    │
    └── [rank 1..N] 从 event.wait() 唤醒
            method_name, args = read_shm()
            self.call(method_name, *args)   ← 与 rank 0 并行执行同样的 run()
            event.clear()
```

---

### 5.4 单次 Forward 的 TP 数据流（完整伪代码，TP=2）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 输入：token_ids (seq_len,)   两个 rank 持有相同副本（replicated）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ EMBEDDING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rank 0: mask=(ids∈[0,75967]);   y0 = embed(safe_ids, W_emb_0); y = mask*y0
rank 1: mask=(ids∈[75968,..]);  y1 = embed(safe_ids, W_emb_1); y = mask*y1
                    ↓ dist.all_reduce(SUM)    ← 通信 ①
       hidden (seq_len, 1024)  [replicated on both ranks]

━━━ DECODER LAYER × 28 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ── RMSNorm (input_layernorm) ─────────────────────────────────────
  rank 0,1: hidden, residual = layernorm(hidden, residual)  [本地]
  hidden: (seq_len, 1024) [replicated]

  ── QKV Projection ────────────────────────────────────────────────
  rank 0: qkv0 = hidden @ W_qkv_0ᵀ   → q0(seq,8,128) k0(seq,4,128) v0(seq,4,128)
  rank 1: qkv1 = hidden @ W_qkv_1ᵀ   → q1(seq,8,128) k1(seq,4,128) v1(seq,4,128)
  [无通信，各 rank 只算自己的 head 片段]

  ── q_norm / k_norm ───────────────────────────────────────────────
  rank 0: q0=q_norm(q0); k0=k_norm(k0)   [本地]
  rank 1: q1=q_norm(q1); k1=k_norm(k1)   [本地]

  ── RoPE ──────────────────────────────────────────────────────────
  rank 0: q0,k0 = rope(positions, q0, k0)  [本地]
  rank 1: q1,k1 = rope(positions, q1, k1)  [本地]

  ── Flash Attention + KV Cache ────────────────────────────────────
  rank 0: store_kvcache(k0,v0 → kv_cache_rank0)   [本地 Triton kernel]
          o0 = flash_attn(q0, k0, v0)              [本地，8 heads]
  rank 1: store_kvcache(k1,v1 → kv_cache_rank1)   [本地 Triton kernel]
          o1 = flash_attn(q1, k1, v1)              [本地，8 heads]
  o0,o1 各: (seq_len, 8, 128) → flatten → (seq_len, 1024)

  ── o_proj (RowParallel) ──────────────────────────────────────────
  rank 0: p0 = o0 @ W_o_0ᵀ   (seq_len, 1024)  ← W_o_0 是 o_proj 的左半列
  rank 1: p1 = o1 @ W_o_1ᵀ   (seq_len, 1024)  ← W_o_1 是 o_proj 的右半列
                    ↓ dist.all_reduce(SUM)    ← 通信 ②（每层）
       attn_out (seq_len, 1024)  [replicated]

  ── RMSNorm (post_attention_layernorm) ────────────────────────────
  rank 0,1: hidden, residual = layernorm(attn_out, residual)  [本地]

  ── gate_up_proj (MergedColumnParallel) ───────────────────────────
  rank 0: gu0 = hidden @ W_gu_0ᵀ  (seq_len, 3072) = [gate0(1536) | up0(1536)]
  rank 1: gu1 = hidden @ W_gu_1ᵀ  (seq_len, 3072) = [gate1(1536) | up1(1536)]
  rank 0: act0 = silu(gate0) * up0   (seq_len, 1536)    [本地]
  rank 1: act1 = silu(gate1) * up1   (seq_len, 1536)    [本地]

  ── down_proj (RowParallel) ───────────────────────────────────────
  rank 0: d0 = act0 @ W_d_0ᵀ   (seq_len, 1024)
  rank 1: d1 = act1 @ W_d_1ᵀ   (seq_len, 1024)
                    ↓ dist.all_reduce(SUM)    ← 通信 ③（每层）
       hidden (seq_len, 1024)  [replicated]

━━━ FINAL NORM + LM HEAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rank 0,1: hidden = final_rmsnorm(hidden)   [本地]
[prefill] hidden = hidden[last_token_indices]   (batch, 1024)

rank 0: logits0 = hidden @ W_lm_0ᵀ   (batch, 75968)
rank 1: logits1 = hidden @ W_lm_1ᵀ   (batch, 75968)
              ↓ dist.gather(dst=rank0)    ← 通信 ④（最终输出）
rank 0: logits = cat([logits0, logits1], dim=-1)   (batch, 151936)
rank 1: returns None

━━━ SAMPLING（仅 rank 0）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
rank 0: token_ids = sampler(logits, temperatures)
rank 1: None   ← 等待下一次来自 rank 0 的 shm 指令
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
通信汇总（每次 forward，28层）：
  all_reduce ×  1  : Embedding
  all_reduce × 28  : Attn o_proj（每层1次）
  all_reduce × 28  : MLP down_proj（每层1次）
  gather     ×  1  : LM Head（最终输出）
  共 57 次通信，均为小张量 (seq_len×1024×2B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 5.5 完整系统时序图

```
时间轴 →
LLMEngine   rank0(GPU0)            rank1(GPU1)           NCCL Bus
   │             │                      │                    │
   │──call()────►│                      │                    │
   │             │──write_shm()         │                    │
   │             │──event.set()────────►│                    │
   │             │                      │◄─event.wait()唤醒  │
   │             │                      │                    │
   │        [Embedding]            [Embedding]               │
   │             │◄─────all_reduce──────►│←────────────────► │
   │             │                      │                    │
   │        [Layer 0..27]          [Layer 0..27]             │
   │    [QKV/Attn/KVcache]    [QKV/Attn/KVcache]            │
   │             │◄──o_proj all_reduce──►│←────────────────► │
   │    [gate_up/SiluMul]    [gate_up/SiluMul]               │
   │             │◄──down  all_reduce───►│←────────────────► │
   │             │         ...（×28）    │                    │
   │         [LM Head]             [LM Head]                  │
   │             │◄────── gather ────────│←────────────────► │
   │             │                      │                    │
   │        [Sampling]                  │（returns None）     │
   │◄──token_ids─│                      │                    │
```

---

## 六、Demo 文件索引

| 文件 | 覆盖内容 |
|------|---------|
| [demo1_nccl_communication.py](demo1_nccl_communication.py) | all_reduce、masked embedding gather、gather logits、barrier |
| [demo2_shared_memory.py](demo2_shared_memory.py) | write_shm/read_shm、Event IPC、pickle 格式、shm 生命周期 |
| [demo3_model_layers.py](demo3_model_layers.py) | ColumnParallel、RowParallel、Merged、QKV、完整 Transformer 层、KV Cache、Vocab/LMHead |

### 运行方式

```bash
# Demo 1 & 3：需要 2 张 GPU（或自动降级到 gloo+CPU）
torchrun --nproc_per_node=2 docs/tp_demos/demo1_nccl_communication.py
torchrun --nproc_per_node=2 docs/tp_demos/demo3_model_layers.py

# Demo 2：纯 CPU，不需要 GPU
python docs/tp_demos/demo2_shared_memory.py
```

## 七、参考文档  
1. [TP并行博客](https://liyuan24.github.io/writings/2025_12_18_nanovllm_tensor_parallel_kernel_fusion.html)
2. [PyTorch Distributed文档](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
3. [NanoVLLM全流程解读](https://zhuanlan.zhihu.com/p/1977336847567983629)
4. [MP文档](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
5. [PyTorch TP教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)


