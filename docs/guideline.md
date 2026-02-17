# NanoVLLM 深度分析指南

本文档由claude code opus4.6生成

## 一、整体流程与项目架构

NanoVLLM 用约 1200 行 Python 实现了 vLLM 的核心推理流程，在 RTX 4070 上跑 Qwen3-0.6B 达到 1434 tok/s（vLLM 为 1362 tok/s）。

### 架构分层

```
┌─────────────────────────────────────────────────┐
│  用户接口层    llm.py / sampling_params.py       │
├─────────────────────────────────────────────────┤
│  引擎调度层    llm_engine.py                     │
│    ├── Scheduler (调度器)                        │
│    ├── BlockManager (KV Cache 页式内存管理)       │
│    └── ModelRunner (GPU 执行器)                  │
├─────────────────────────────────────────────────┤
│  算子层        layers/                           │
│    ├── attention.py   (FlashAttn + PagedKV)     │
│    ├── linear.py      (张量并行线性层)            │
│    ├── sampler.py     (Gumbel-max 采样)          │
│    ├── rotary_embedding.py / layernorm.py / ... │
├─────────────────────────────────────────────────┤
│  模型层        models/qwen3.py                   │
├─────────────────────────────────────────────────┤
│  工具层        utils/context.py, loader.py       │
└─────────────────────────────────────────────────┘
```

### 项目文件结构

```
nanovllm/
├── engine/                    # 推理引擎核心
│   ├── llm_engine.py         # 总协调器，驱动整个推理流程
│   ├── scheduler.py          # 智能调度器，决定执行顺序
│   ├── block_manager.py      # KV缓存内存管理 (PagedAttention核心)
│   ├── model_runner.py       # 单GPU上的模型执行器
│   └── sequence.py           # 请求序列的数据结构
├── layers/                    # 神经网络层实现
│   ├── attention.py          # FlashAttention + KV缓存管理
│   ├── sampler.py            # 从logits采样生成token
│   ├── linear.py             # 支持张量并行的线性层
│   ├── layernorm.py          # RMS LayerNorm
│   ├── rotary_embedding.py   # 旋转位置编码 (RoPE)
│   ├── activation.py         # 激活函数 (SiLU)
│   └── embed_head.py         # 词嵌入和语言模型头
├── models/                    # 具体模型架构
│   └── qwen3.py              # Qwen3模型完整实现
├── utils/                     # 工具模块
│   ├── context.py            # 全局上下文状态管理
│   └── loader.py             # 模型权重加载器
├── config.py                 # 配置管理
├── llm.py                    # 用户接口入口
└── sampling_params.py        # 采样参数定义
```

### 请求全生命周期

```
用户 prompt
  → tokenize → 创建 Sequence 对象 → 加入 Scheduler.waiting 队列
  → 主循环:
      1. Scheduler.schedule()
         ├─ waiting 非空 → Prefill: 分配 block, 移入 running
         └─ waiting 为空 → Decode: 检查显存, 必要时抢占
      2. ModelRunner.run(seqs, is_prefill)
         ├─ prepare_prefill() / prepare_decode()
         │   → 构造 input_ids, positions, slot_mapping, block_tables
         ├─ set_context() → 全局上下文供 Attention 层读取
         ├─ run_model() → 前向传播 (eager 或 CUDA Graph replay)
         └─ sampler() → 采样下一个 token (仅 rank 0)
      3. Scheduler.postprocess()
         ├─ 追加 token 到 Sequence
         ├─ 检查终止条件 (EOS / max_tokens)
         └─ 终止的序列 deallocate + FINISHED
  → detokenize → 返回结果
```

核心设计决策：prefill 和 decode 不混合调度，prefill 优先。这简化了实现，同时保证了 TTFT（Time To First Token）。

### 各组件职责划分

| 组件 | 职责 |
| :--- | :--- |
| LLMEngine | 主循环，协调调度和执行 |
| Scheduler | 决定每一步执行哪些请求 |
| BlockManager | 页式 KV 缓存的分配与回收 |
| ModelRunner | 准备输入、执行模型、处理输出 |
| Attention | 集成 Flash Attention 和 KV Cache 写入 |
| 并行层 | 支持张量并行的线性层和 Embedding |

---

## 二、核心技术点详解

### 1. PagedAttention（页式 KV Cache）

将 KV Cache 按固定大小（默认 256 tokens）分块管理，类似操作系统虚拟内存的分页机制。

**核心机制：**
- 每个 Sequence 维护一个 `block_table`（逻辑块 → 物理块的映射）
- BlockManager 维护 `free_block_ids` / `used_block_ids`，按需分配
- `slot_mapping` 将 token 位置映射到物理 cache 位置：`block_id * block_size + offset`
- 解决了传统 KV Cache 预分配最大长度导致的显存浪费和碎片化问题

**KV Cache 物理存储布局：**
```python
# 系统启动时预分配
Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
#       K/V    L          B          T           H            D

# 每个 Attention 层获得一个 view
module.k_cache = kv_cache[0, layer_id]  # [B, T, H, D]
module.v_cache = kv_cache[1, layer_id]
```

**实现位置：** `nanovllm/engine/block_manager.py`、`nanovllm/layers/attention.py`

**参考资料：**
- 论文：[Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM, SOSP'23)](https://arxiv.org/abs/2309.06180)
- 博客：[vLLM PagedAttention 原理](https://blog.vllm.ai/2023/06/20/vllm.html)

### 2. Prefix Caching（前缀缓存）

多个请求共享相同前缀（如 system prompt）时，复用已计算的 KV Cache block。

**核心机制：**
- 对每个满块（256 tokens）计算 hash（使用 xxhash，链式 hash 包含前一块的 hash）
- `hash_to_block_id` 字典做查找，命中则 `ref_count++`，跳过计算
- 只有满块才参与 hash，最后一个不满的块不缓存
- 释放时 `ref_count--`，降到 0 才真正回收

**工作流程：**
```python
# allocate() 中的前缀缓存逻辑
for each logical block in sequence:
    1. 计算 hash（仅满块，包含前一块 hash 的链式 hash）
    2. 查找 hash_to_block_id
    3. 命中 → 复用 block, ref_count++, 更新 num_cached_tokens
    4. 未命中 → 从 free list 分配新 block, 存储 hash 映射
    5. 追加 block_id 到 seq.block_table
```

**实现位置：** `nanovllm/engine/block_manager.py` 的 `allocate()` 方法

**参考资料：**
- vLLM 的 Automatic Prefix Caching 文档
- SGLang 的 RadixAttention：[论文](https://arxiv.org/abs/2312.07104)

### 3. Flash Attention

NanoVLLM 直接调用 `flash-attn` 库，分两种场景：

**Prefill 阶段：**
- 使用 `flash_attn_varlen_func` — 处理变长序列的 batch
- 通过 `cu_seqlens_q/k` 标记每个序列的边界
- 有 prefix cache 时，`seqlen_q < seqlen_k`，通过 `block_table` 访问已缓存的 KV

**Decode 阶段：**
- 使用 `flash_attn_with_kvcache` — 每个序列只有 1 个 query token
- 通过 `block_table` 访问非连续的 paged KV Cache
- 通过 `cache_seqlens` 指定每个序列的上下文长度

**KV Cache 写入：**
- Triton kernel `store_kvcache_kernel` 负责将新计算的 K/V 写入物理 cache 位置

**实现位置：** `nanovllm/layers/attention.py`

**参考资料：**
- 论文：[FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al.)](https://arxiv.org/abs/2205.14135)
- 论文：[FlashAttention-2](https://arxiv.org/abs/2307.08691)
- 项目：[github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

### 4. CUDA Graph

Decode 阶段每步只处理 1 个 token/seq，kernel launch 开销占比大。CUDA Graph 将整个计算图录制下来，后续直接 replay。

**核心机制：**
- 预录制多个 batch size 的 graph（1, 2, 4, 8, ..., 512）
- 运行时选择 ≥ 实际 batch size 的最小 2 的幂次
- 将实际输入 copy 到 graph 的 placeholder tensor，然后 replay
- 仅用于 decode 阶段（prefill 的 shape 变化太大，不适合录制）

**执行逻辑：**
```python
if is_prefill or enforce_eager or batch_size > 512:
    # Eager 模式：直接执行
    hidden = model(input_ids, positions)
    logits = model.compute_logits(hidden)
else:
    # CUDA Graph 模式：replay 预录制的图
    graph = graphs[next_power_of_2(batch_size)]
    copy inputs to graph_vars
    graph.replay()
    logits = model.compute_logits(graph_vars["outputs"][:batch_size])
```

**实现位置：** `nanovllm/engine/model_runner.py` 的 `capture_cudagraph()` 和 `run_model()`

**参考资料：**
- NVIDIA 文档：[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- PyTorch 文档：[torch.cuda.CUDAGraph](https://pytorch.org/docs/stable/cuda.html)
- 博客：[Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

### 5. Tensor Parallelism（张量并行）

将模型权重切分到多张 GPU 上，采用 Megatron-LM 风格的列并行 + 行并行组合。

**切分策略：**

| 层 | 切分方式 | 通信 |
|---|---|---|
| QKV Projection | 列并行（按 head 切） | 无 |
| O Projection | 行并行（按输入维切） | all_reduce |
| Gate+Up Proj | 列并行（合并） | 无 |
| Down Proj | 行并行 | all_reduce |
| Embedding | 按词表切 | all_reduce |
| LM Head | 按词表切 | gather 到 rank 0 |

**列并行（ColumnParallel）：**
```
W = [W_0 | W_1 | ... | W_{n-1}]   每个 rank 持有 W_i
每个 rank 计算: Y_i = X @ W_i
最终输出: Y = [Y_0 | Y_1 | ... | Y_{n-1}]  (沿特征维拼接)
无需通信
```

**行并行（RowParallel）：**
```
W = [W_0; W_1; ...; W_{n-1}]   每个 rank 持有 W_i
输入也对应切分: X = [X_0 | X_1 | ... | X_{n-1}]
每个 rank 计算: Y_i = X_i @ W_i
最终输出: Y = sum(Y_0, ..., Y_{n-1})  (需要 all_reduce)
```

**多卡 IPC 通信：**
- NCCL 做张量通信（all_reduce / gather）
- Shared Memory 做控制信令（方法名 + 参数的序列化传输）
- Multiprocessing Event 做同步

**实现位置：** `nanovllm/layers/linear.py`、`nanovllm/layers/embed_head.py`

**参考资料：**
- 论文：[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- 项目：[github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

### 6. Continuous Batching（连续批处理）

不同于静态 batching（等所有序列结束才处理下一批），Scheduler 在每一步动态调整 batch 组成。

**核心机制：**
- 完成的序列立即释放资源
- 新请求可以随时加入 waiting 队列
- 抢占机制：显存不足时，LIFO 抢占 running 中的序列，释放其 block 后重新排队

**调度逻辑：**
```
schedule() 被调用
    ├─ waiting 非空？
    │   ├─ 是：尝试 Prefill admit
    │   │       逐个检查 waiting 队首
    │   │       满足约束则 allocate + RUNNING + 移入 running
    │   │       返回 (scheduled, is_prefill=True)
    │   └─ 否：进入 Decode
    │           逐个处理 running
    │           需要新块但没有？抢占其他请求或自抢占
    │           may_append
    │           返回 (scheduled, is_prefill=False)
```

**实现位置：** `nanovllm/engine/scheduler.py`

**参考资料：**
- 论文：[Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu)

### 7. torch.compile + Triton

对 RMSNorm、RotaryEmbedding、Sampler 使用 `@torch.compile`，PyTorch 编译器自动做算子融合并生成 Triton kernel。

**应用场景：**
- RMSNorm：fused add + norm，避免多次显存读写
- RotaryEmbedding：cos/sin 计算与旋转融合
- Sampler：temperature scaling + softmax + Gumbel-max 采样融合

**参考资料：**
- [PyTorch 2.0 torch.compile 文档](https://pytorch.org/docs/stable/torch.compiler.html)
- [OpenAI Triton](https://github.com/triton-lang/triton)

---

## 三、对比 vLLM：NanoVLLM 省略了哪些工程细节

| 维度 | vLLM | NanoVLLM | 影响 |
|---|---|---|---|
| 模型支持 | 100+ 模型架构（LLaMA, Mistral, GPT, 多模态等） | 仅 Qwen3 | 不可直接用于其他模型 |
| 采样策略 | top-k, top-p, beam search, min-p, repetition penalty, logits processor 等 | 仅 temperature sampling（Gumbel-max） | 无法做复杂生成控制 |
| 量化 | AWQ, GPTQ, FP8, INT8, GGUF, Marlin kernel 等 | 无量化，仅 FP16/BF16 | 无法在小显存上跑大模型 |
| Speculative Decoding | 支持 draft model / Medusa / Eagle 等 | 无 | 缺少加速长序列生成的手段 |
| 在线服务 | 完整 OpenAI 兼容 API server（AsyncLLMEngine + FastAPI） | 仅离线 batch 推理 | 无法作为服务部署 |
| 异步引擎 | AsyncLLMEngine，请求级别异步 | 同步阻塞式 `generate()` | 无法流式输出、无法动态添加请求 |
| Pipeline Parallelism | 支持 PP（跨层切分） | 仅 TP（层内切分） | 无法跨节点扩展超大模型 |
| Chunked Prefill | 将长 prefill 拆分成多个 chunk，与 decode 交错执行 | prefill 一次性处理，且 prefill/decode 不混合 | 长 prompt 会阻塞 decode，影响延迟 |
| Preemption 策略 | Swap（KV Cache 换出到 CPU）+ Recompute | 仅 Recompute（deallocate 后重新计算） | 抢占代价更高 |
| KV Cache 管理 | 支持 CPU offload、disk offload、自动 eviction 策略 | 仅 GPU 内存，简单 free list | 显存受限时灵活性差 |
| 调度算法 | 多种策略（priority, fairness, SLA-aware） | 简单 FIFO + prefill 优先 | 无法保证公平性和 SLA |
| 分布式 | Ray 集成，支持多节点多卡 | 单机多卡（multiprocessing + shared memory） | 无法跨机器扩展 |
| LoRA | 动态 LoRA 加载，多 LoRA 并发服务 | 无 | 无法做多租户微调模型服务 |
| Structured Output | JSON schema 约束生成、正则引导 | 无 | 无法保证输出格式 |
| 前缀缓存 | 自动 + 手动，支持 eviction 策略、LRU 等 | 基础 hash 匹配，无 eviction 策略 | 缓存满后无法智能淘汰 |
| 监控 & 可观测性 | Prometheus metrics、详细日志、tracing | 仅 Nsight profiling 支持 | 生产环境缺少监控手段 |
| 错误处理 | 完善的异常处理、请求重试、graceful degradation | 最小化错误处理 | 生产环境不够健壮 |
| 自定义 kernel | 大量手写 CUDA kernel（PagedAttention kernel、fused ops 等） | 依赖 flash-attn 库 + 1 个 Triton kernel | 灵活性和极致优化空间受限 |

---

## 四、关键配置参数说明

| 参数 | 默认值 | 影响 | 权衡 |
|---|---|---|---|
| `max_num_batched_tokens` | 16384 | Prefill 吞吐上限 | 太大可能 OOM |
| `max_num_seqs` | 512 | 并发度上限，影响 Decode 吞吐 | 太大增加显存压力 |
| `max_model_len` | 4096 | 最大序列长度 | 影响 CUDA Graph 捕获时的 block_tables 大小 |
| `gpu_memory_utilization` | 0.9 | KV Cache 块数 | 太高容易 OOM |
| `kvcache_block_size` | 256 | 块粒度 | 大块减少元数据开销，但降低缓存命中率 |
| `enforce_eager` | False | 是否禁用 CUDA Graph | True 便于调试，但推理更慢 |
| `tensor_parallel_size` | 1 | 张量并行 GPU 数 | 更多 GPU 支持更大模型，但增加通信开销 |

---

## 五、总结

NanoVLLM 是一个极好的教学项目——它用最精简的代码实现了 LLM 推理引擎的核心骨架：

1. **PagedAttention** — 页式 KV Cache 内存管理
2. **Prefix Caching** — 共享前缀优化
3. **CUDA Graph** — 消除 kernel launch 开销
4. **Tensor Parallelism** — 多 GPU 张量并行
5. **Continuous Batching** — 动态连续批处理
6. **Flash Attention** — 高效注意力计算
7. **torch.compile** — 算子融合优化

它省略了 vLLM 作为生产系统所需的大量工程细节（多模型适配、丰富的采样/量化/并行策略、在线服务能力、异步架构、以及生产级的容错与可观测性），但保留了理解 LLM 推理引擎所需的全部核心概念。

---

## 参考资料

### 论文
- [Efficient Memory Management for Large Language Model Serving with PagedAttention (SOSP'23)](https://arxiv.org/abs/2309.06180)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI'22)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [SGLang RadixAttention](https://arxiv.org/abs/2312.07104)

### 项目
- [vLLM](https://github.com/vllm-project/vllm)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [SGLang](https://github.com/sgl-project/sglang)
- [OpenAI Triton](https://github.com/triton-lang/triton)

### 博客
- [vLLM PagedAttention 原理](https://blog.vllm.ai/2023/06/20/vllm.html)
- [nano-vLLM 学习后记](https://zhuanlan.zhihu.com/p/1977336847567983629)
- [NanoVLLM Part 1](https://neutree.ai/blog/nano-vllm-part-1)
- [NanoVLLM Part 2](https://neutree.ai/blog/nano-vllm-part-2)
- [vLLM 入门文档](https://www.aleksagordic.com/blog/vllm)
