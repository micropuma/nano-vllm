# NanoVLLM 后续优化 Todo & 技术路线

> 目标：围绕 NanoVLLM 当前架构（`LLMEngine -> Scheduler -> ModelRunner -> Layers`）构建可持续演进路线。
> 
> 覆盖五个方向：
> 1) 更多模型支持 2) 算子/Kernel Fusion 3) 精度与压缩 4) 推理技术（含 Chunked Prefill）5) Benchmark 体系（含 TTFT/TPOT）

---

## 0. 当前基线（便于后续衡量）

### 已有能力
- Continuous batching（prefill/decode 分离调度）
- Paged KV Cache + Prefix Cache（`BlockManager` + `slot_mapping`）
- CUDA Graph（decode 路径）
- Tensor Parallel（多进程 + shared memory 控制）
- FlashAttention + Triton KV 写入

### 关键代码锚点
- 调度：`nanovllm/engine/scheduler.py`
- KV 管理：`nanovllm/engine/block_manager.py`
- 执行器：`nanovllm/engine/model_runner.py`
- 引擎入口：`nanovllm/engine/llm_engine.py`
- 注意力与算子：`nanovllm/layers/attention.py`
- 模型实现（当前仅 Qwen3）：`nanovllm/models/qwen3.py`
- Benchmark：`bench.py`

---

## 1) 更多模型支持

### 1.1 目标
- 从“单模型（Qwen3）可跑”升级到“多 HuggingFace CausalLM 架构可跑”。
- 优先覆盖：Llama 系、Qwen2/Qwen2.5、Mistral；后续再到 Mixtral（MoE）。

### 1.2 技术路线

#### 路径 A：按架构手写高性能实现（推荐短中期）
- 新增模型文件：
  - `nanovllm/models/llama.py`
  - `nanovllm/models/mistral.py`
  - （可选）`nanovllm/models/qwen2.py`
- 对齐统一接口：
  - `forward(input_ids, positions)`
  - `compute_logits(hidden_states)`
  - `packed_modules_mapping`（用于权重加载映射）
- 复用现有并行层：`QKVParallelLinear` / `RowParallelLinear` / `MergedColumnParallelLinear`

#### 路径 B：先做“兼容优先”的通用 HF 适配层（推荐并行推进）
- 新增 `nanovllm/models/registry.py`，做 model_type -> class 映射。
- 在 `ModelRunner` 里根据 `config.hf_config.model_type` 选择模型类。
- 新增 fallback 路径（未适配模型给出清晰错误提示）。

#### 路径 C：MoE 扩展（中长期）
- 支持 Mixtral/DeepSeek-MoE 类结构：
  - Router Top-k 选择
  - expert 并行策略（先单机单卡，再 TP）
  - 稀疏激活下的 dispatch kernel

### 1.3 可能修改文件
- 修改：
  - `nanovllm/engine/model_runner.py`（模型构造与选择）
  - `nanovllm/utils/loader.py`（权重名映射扩展）
  - `nanovllm/config.py`（模型能力检查、特性开关）
- 新增：
  - `nanovllm/models/registry.py`
  - `nanovllm/models/llama.py`
  - `nanovllm/models/mistral.py`

### 1.4 验收指标
- 同一 API 下能切换至少 2 个非 Qwen3 模型并稳定生成。
- 与 HF reference 在小 batch 下输出一致性通过（容忍采样波动）。

---

## 2) 算子层面优化（CUDA/Triton + Kernel Fusion）

### 2.1 目标
- 降低 decode 单步 latency，提升吞吐并降低显存带宽压力。

### 2.2 技术路线

#### 路径 A：Triton 内核持续替换 Python 热点
- 当前已有 `store_kvcache_kernel`，可继续扩展：
  - block_table gather / slot transform 的预处理 kernel
  - dequant + matmul 融合前处理 kernel（配合量化）
- 对 `prepare_decode`/`prepare_prefill` 中 CPU 侧张量拼接做 CUDA 化或 pinned-memory 优化。

#### 路径 B：Kernel Fusion（优先高收益链路）
- 候选 fusion：
  1. `RMSNorm + QKVLinear`（减少中间读写）
  2. `RoPE + Q/K reshape`（减少格式转换）
  3. `SiLU + Mul`（已部分融合，可继续与上游投影衔接）
- 对比 eager/torch.compile/Triton 自定义 kernel 三条路径收益。

#### 路径 C：更深层图优化
- `torch.compile` 按 prefill/decode 分别策略化开启。
- CUDA Graph 覆盖扩展：目前 decode 主路径，探索 prefill 小批场景可行性。

### 2.3 可能修改文件
- 修改：
  - `nanovllm/layers/attention.py`
  - `nanovllm/layers/layernorm.py`
  - `nanovllm/layers/linear.py`
  - `nanovllm/layers/activation.py`
  - `nanovllm/engine/model_runner.py`（graph 与 input staging）
- 新增（建议）：
  - `nanovllm/layers/kernels/`（集中管理 triton/cuda kernels）

### 2.4 验收指标
- Decode token latency 下降（如 p50/p95）。
- 单卡 tokens/s 提升，且显存占用不显著上升。

---

## 3) 精度/压缩（量化、KV Cache 压缩、Sparse KV）

### 3.1 目标
- 在可接受质量损失下换取更高吞吐与更长上下文。

### 3.2 技术路线

#### 路径 A：权重量化（先易后难）
- 阶段 1：W8A16（weight-only int8）
- 阶段 2：W4A16（GPTQ/AWQ 风格，离线量化 + 在线反量化）
- 阶段 3：W8A8 或 FP8（需要更系统校准）

落地点：
- 线性层增加 quantized weight 类型与加载逻辑。
- 权重 loader 支持量化权重格式。

#### 路径 B：KV Cache 压缩
- KV int8/int4 cache（分通道或分头 scale）
- decode 前按需反量化参与 attention
- 可配合 residual cache（最近 token 保留高精）

#### 路径 C：Sparse KV Cache（探索项）
- 局部窗口 + 全局 token 选择（如 sink tokens）
- 查询相关性选择性保留 KV block
- 与 paged cache 配合实现“块级稀疏”

### 3.3 可能修改文件
- 修改：
  - `nanovllm/layers/linear.py`（量化线性层）
  - `nanovllm/utils/loader.py`（量化权重读取）
  - `nanovllm/engine/model_runner.py`（KV cache dtype/布局）
  - `nanovllm/layers/attention.py`（KV 反量化、稀疏读取）
  - `nanovllm/config.py`（新增量化/压缩配置）
- 新增（建议）：
  - `nanovllm/quantization/`（量化策略与算子封装）

### 3.4 验收指标
- 在固定质量阈值下（如 ppl 或任务分数下降 < 目标值），
  - 显存降低
  - 可服务上下文长度提升
  - 吞吐有净增益

---

## 4) 推理技术支持（含 Chunked Prefill）

### 4.1 目标
- 提升混合负载（长 prompt + 高频短请求）下 tail latency 与稳定性。

### 4.2 技术路线

#### 路径 A：Chunked Prefill（优先级最高）
- 将长 prompt 切分为固定 chunk（如 256/512 tokens）逐步 prefill。
- decode 与 prefill chunk 可交替调度，避免长请求独占 GPU。
- 需要 scheduler 支持“同一 sequence 的分段 prefill 状态机”。

#### 路径 B：Prefill/Decode 混部调度
- 从当前“严格分离”升级到“受控混合”：
  - 每步给 decode 保底 token 预算
  - prefill 使用剩余 token budget
- 引入简单 cost model（按 step latency 估算）动态调参。

#### 路径 C：高级调度策略
- SLO-aware（按 deadline 或优先级）
- preemption 策略从“recompute only”到“recompute + selective retain”

### 4.3 可能修改文件
- 修改：
  - `nanovllm/engine/sequence.py`（新增 chunk 进度字段）
  - `nanovllm/engine/scheduler.py`（chunk/mixed scheduling 主逻辑）
  - `nanovllm/engine/model_runner.py`（prefill 输入组装支持分段）
  - `nanovllm/engine/llm_engine.py`（统计与可观测性）
  - `nanovllm/config.py`（chunk size、mix ratio 等配置）

### 4.4 验收指标
- 长 prompt 场景下 TTFT p95 降低。
- 混合流量场景下短请求 TPOT 更稳定。

---

## 5) Benchmark 体系（补齐 TTFT / TPOT / 分位数）

### 5.1 目标
- 从“仅总吞吐”升级到“可指导优化决策”的指标体系。

### 5.2 技术路线

#### 路径 A：最小可用指标（第一阶段）
- 在 `bench.py` 增加：
  - TTFT（Time To First Token）
  - TPOT（Time Per Output Token）
  - e2e latency（per request）
  - 吞吐（input/output/total tokens per sec）

#### 路径 B：分位数与可视化（第二阶段）
- 输出 p50/p90/p95/p99。
- 保存 JSON/CSV 结果用于多次实验对比。
- 与 `trace.py` 打通，支持 profiling 结果关联。

#### 路径 C：场景化压测（第三阶段）
- 构建 workload profile：
  - 短 prompt + 长输出
  - 长 prompt + 短输出
  - 混合 burst 流量
- 对比 eager / cudagraph / tp / quant 等开关矩阵。

### 5.3 可能修改文件
- 修改：
  - `bench.py`
  - `nanovllm/engine/llm_engine.py`（暴露 step 级 timing hooks）
  - `trace.py`（实验标签、trace 输出结构）
- 新增（建议）：
  - `nanovllm/utils/metrics.py`
  - `scripts/bench_matrix.py`

### 5.4 指标定义建议
- `TTFT`：请求提交到首个 token 产出的时延。
- `TPOT`：
  $$TPOT = \frac{t_{last\_token} - t_{first\_token}}{N_{output\_tokens}-1}$$
- `Throughput`：
  $$\text{tok/s} = \frac{\sum N_{tokens}}{\text{wall clock time}}$$

---

## 6) 分阶段执行建议（优先级）

### P0（1~2 周）
- [ ] `bench.py` 补齐 TTFT/TPOT/分位数
- [ ] 引入 model registry + 至少 1 个新模型（Llama 或 Mistral）
- [ ] chunked prefill 设计文档与最小实现（不做混部）

### P1（2~4 周）
- [ ] chunked prefill 正式版本 + mixed scheduling v1
- [ ] kernel fusion 第一批（RMSNorm+QKV 或 RoPE 相关）
- [ ] W8A16 weight-only 量化打通

### P2（4~8 周）
- [ ] KV cache int8 压缩
- [ ] sparse KV 原型
- [ ] 多模型矩阵 benchmark + 自动化回归

---

## 7) 任务追踪模板（可复制）

```markdown
- [ ] 任务名：
  - 目标：
  - 技术方案：
  - 代码改动：
    - 修改：
    - 新增：
  - 验收指标：
  - 风险与回滚：
```

---

## 8) 建议先做的三个“高 ROI”任务

1. **Benchmark 指标补齐（TTFT/TPOT/p95）**
   - 快速建立优化反馈闭环，后续所有工作都依赖它。
2. **Chunked Prefill（最小版本）**
   - 对真实服务场景 tail latency 改善明显。
3. **Llama 模型支持 + Registry**
   - 提升项目泛化能力，方便对比不同结构上的优化收益。
