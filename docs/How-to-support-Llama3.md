# How to support Llama3 in NanoVLLM

本文档说明如何在 NanoVLLM 中支持 Llama3/Llama3.1 系列模型，以及与已有 Qwen3 实现的架构差异。

---

## 架构对比：Llama3 vs Qwen3

### 总览

| 维度 | Qwen3 | Llama3 / Llama3.1 |
|---|---|---|
| HF architecture 字段 | `Qwen3ForCausalLM` | `LlamaForCausalLM` |
| QKV bias | 可配置（`config.attention_bias`） | 固定 `False` |
| Q/K Head Norm | ✅ `q_norm` + `k_norm`（每 head 做 RMSNorm） | ❌ 无 |
| RoPE theta | `1,000,000` | `500,000`（Llama3.1 通常在 config 里显式指定） |
| RoPE Scaling | 通常 `null` | Llama3.1+ 使用 `"rope_type": "llama3"` 高低频混合缩放 |
| `tie_word_embeddings` | `True`（小模型） | `False`（独立 lm_head） |
| MLP 结构 | SwiGLU（gate_up + down） | 完全相同 |
| Norm 类型 | RMSNorm | 完全相同 |
| 残差连接 | fused add-norm | 完全相同 |

---

## 核心差异详解

### 1. Q/K Head Normalization

**Qwen3** 在 QKV projection 之后对每个 attention head 的 Q 和 K 分别做 RMSNorm：

```python
# qwen3.py — Qwen3Attention
if not self.qkv_bias:
    self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

# forward 中：
q = self.q_norm(q)
k = self.k_norm(k)
```

**Llama3** 没有这两个 norm，RoPE 直接作用于原始的 Q/K：

```python
# llama3.py — Llama3Attention
# forward 中（无 q_norm/k_norm）：
q, k = self.rotary_emb(positions, query, key)
```

> **为什么 Qwen3 需要？** Qwen3 在大规模训练时发现 attention logit 数值容易发散，Head Norm 可以稳定训练。Llama3 通过其他训练技巧（如更保守的 lr 调度）避免了这个问题。

---

### 2. RoPE Scaling（Llama3.1+ 长上下文）

**Qwen3**：`rope_scaling = null`，直接使用原始 RoPE（`RotaryEmbedding`）。

**Llama3**（base）：同样无 scaling，`rope_scaling = null`。

**Llama3.1**（128K 上下文）：使用高低频混合缩放，config 中：

```json
{
  "rope_scaling": {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192
  }
}
```

缩放逻辑（`Llama3RotaryEmbedding`，位于 `layers/rotary_embedding.py`）：

```
wavelen = 2π / freq

if wavelen < high_freq_wavelen:    # 高频（短波长）→ 不缩放
    pass
elif wavelen > low_freq_wavelen:   # 低频（长波长）→ 除以 factor
    scaled_freq = freq / factor
else:                              # 中间区域 → smooth blend
    smooth = (original_max_pos/wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    scaled_freq = (1-smooth)*freq/factor + smooth*freq
```

> **lru_cache 问题**：`get_rope` 使用 `@lru_cache`，要求所有参数可哈希。`dict` 不可哈希，因此调用处需将 `rope_scaling` 转为 `tuple(sorted(items()))`，`get_rope` 内部再转回 `dict`。

---

### 3. QKV Bias

**Qwen3** 的 `attention_bias` 在不同规格的模型里不一样（有些版本为 True）：

```python
# qwen3.py
self.qkv_proj = QKVParallelLinear(..., bias=qkv_bias)  # 从 config 读取
```

**Llama3** 固定无 bias：

```python
# llama3.py
self.qkv_proj = QKVParallelLinear(..., bias=False)  # 硬编码
```

---

### 4. lm_head 的位置

**Qwen3**：`lm_head` 挂在 `Qwen3ForCausalLM`（顶层），`Qwen3Model` 不含 `lm_head`。

**Llama3**（本实现）：同样将 `lm_head` 放在 `Llama3ForCausalLM`，保持一致：

```python
class Llama3ForCausalLM(nn.Module):
    def __init__(self, config):
        self.model = Llama3Model(config)        # 不含 lm_head
        self.lm_head = ParallelLMHead(...)      # 这里初始化
```

> ⚠️ 注意：`lm_head` 必须先于 `tie_word_embeddings` 赋值初始化，否则会 `AttributeError`。

---

## 新增/修改的文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `nanovllm/models/llama3.py` | 新增 | Llama3 完整前向实现 |
| `nanovllm/layers/rotary_embedding.py` | 修改 | 新增 `Llama3RotaryEmbedding`，`get_rope` 支持 `rope_type: "llama3"`，入参改为 `tuple \| None` 以支持 `lru_cache` |
| `nanovllm/engine/model_runner.py` | 修改 | 移除硬编码 `Qwen3ForCausalLM`，改为 `_ARCHITECTURE_MAP` 动态派发 |

---

## 模型加载：packed_modules_mapping

HF checkpoint 中 Q/K/V 是三个独立的权重矩阵（`q_proj`/`k_proj`/`v_proj`），而本项目将它们合并为一个 `qkv_proj`（减少 3 次 matmul 为 1 次）。`loader.py` 通过 `packed_modules_mapping` 自动处理这个映射，Llama3 与 Qwen3 的 mapping **完全相同**：

```python
packed_modules_mapping = {
    "q_proj":    ("qkv_proj", "q"),
    "k_proj":    ("qkv_proj", "k"),
    "v_proj":    ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj":   ("gate_up_proj", 1),
}
```

---

## 使用方式

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Llama-3.1-8B-Instruct", enforce_eager=False, tensor_parallel_size=1)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, who are you?"], sampling_params)
print(outputs[0]["text"])
```

`model_runner.py` 会自动读取 `config.json` 中的 `"architectures": ["LlamaForCausalLM"]`，路由到 `Llama3ForCausalLM`，无需任何额外配置。

---

## 扩展 Llama3 变体（EAGLE 等）

`Llama3Model` 和 `Llama3ForCausalLM` 预留了 `layer_type` 参数，允许替换 `Llama3DecoderLayer` 为自定义子类：

```python
class DraftDecoderLayer(Llama3DecoderLayer):
    # 省略部分 sublayer 用于 draft model
    ...

llm_causal = Llama3ForCausalLM(config, layer_type=DraftDecoderLayer)
```

---

## 支持新模型的通用步骤

1. 在 `nanovllm/models/` 下新建 `xxx.py`，实现 `XxxForCausalLM`（含 `packed_modules_mapping`、`forward`、`compute_logits`）
2. 在 `model_runner.py` 的 `_ARCHITECTURE_MAP` 中添加一行映射
3. 若使用新的 RoPE 变体，在 `rotary_embedding.py` 中添加对应类并在 `get_rope` 中注册
