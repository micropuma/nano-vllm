"""
Demo 3: NanoVLLM 模型层面的 Tensor Parallelism
================================================

本文件覆盖 NanoVLLM 中所有涉及 TP 的模型层，完整复现其 forward 和 weight_loader 逻辑：

  1. ColumnParallelLinear    — 按输出维度切分（Column）
  2. RowParallelLinear       — 按输入维度切分（Row）+ all_reduce
  3. MergedColumnParallelLinear — gate_proj + up_proj 合并列并行
  4. QKVParallelLinear       — Q/K/V 按注意力头切分
  5. VocabParallelEmbedding  — vocab 表按词表切分
  6. ParallelLMHead          — LM head + gather logits
  7. Attention + MLP 完整 TP 数据流

Megatron-LM 风格 TP 的核心不变量：
  Column Parallel  → output 维度 ÷ tp_size，不需要通信，下游接 Row Parallel
  Row Parallel     → input  维度 ÷ tp_size，需要 all_reduce 汇聚部分和

运行方式（需要 2 张 GPU，否则自动降级为 gloo+CPU 模式）：
  torchrun --nproc_per_node=2 demo3_model_layers.py
  # 或
  python demo3_model_layers.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def setup(rank, world_size, backend="nccl"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def divide(a, b):
    assert a % b == 0, f"{a} % {b} != 0"
    return a // b


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    dist.broadcast(t, src=src)
    return t


# ─────────────────────────────────────────────────────────────────────────────
# 1. ColumnParallelLinear
# ─────────────────────────────────────────────────────────────────────────────
# 对应：nanovllm/layers/linear.py  ColumnParallelLinear
#
# 切分方式：
#   完整权重 W ∈ R^{out × in}
#   rank i 持有 W[i*shard : (i+1)*shard, :]，shard = out // tp_size
#
# 数据流：
#   x (BATCH, in) —[每 rank 各自 F.linear]→ y_i (BATCH, out/tp)   # 无需通信!
#
# 不变量：
#   y_i 只是完整输出的一段，通常直接传给 RowParallelLinear（它需要分片输入）
# ─────────────────────────────────────────────────────────────────────────────

class ColumnParallelLinear(nn.Module):
    """复刻 nanovllm/layers/linear.py:ColumnParallelLinear"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.shard_out = divide(out_features, self.tp_size)
        self.weight = nn.Parameter(torch.empty(self.shard_out, in_features))
        nn.init.normal_(self.weight, std=0.02)

    def weight_loader(self, full_weight: torch.Tensor):
        """
        从完整权重按 dim=0（输出维度）切分，每个 rank 拿自己的一段。
        对应 ColumnParallelLinear.weight_loader() 中的 narrow(tp_dim=0, ...)
        """
        shard = full_weight.narrow(0, self.tp_rank * self.shard_out, self.shard_out)
        self.weight.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ColumnParallel 不需要通信：每个 rank 独立计算自己负责的输出列
        return F.linear(x, self.weight)          # (BATCH, shard_out)


def demo_column_parallel(rank, world_size, device):
    """验证 ColumnParallelLinear 输出等价于串行完整计算的对应切片。"""
    BATCH, IN, OUT = 4, 8, 16

    # 广播同一份完整权重和输入
    full_w = broadcast_tensor(torch.randn(OUT, IN, device=device))
    x      = broadcast_tensor(torch.randn(BATCH, IN, device=device))

    layer = ColumnParallelLinear(IN, OUT).to(device)
    layer.weight_loader(full_w)

    y_shard = layer(x)   # (BATCH, OUT/tp)

    # 串行参考
    y_full_ref = x @ full_w.T                         # (BATCH, OUT)
    shard_size = OUT // world_size
    y_ref_shard = y_full_ref[:, rank * shard_size : (rank + 1) * shard_size]

    assert torch.allclose(y_shard, y_ref_shard, atol=1e-5)
    log(rank, f"[1] ColumnParallelLinear  ✓  shard shape={list(y_shard.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RowParallelLinear
# ─────────────────────────────────────────────────────────────────────────────
# 对应：nanovllm/layers/linear.py  RowParallelLinear
#
# 切分方式：
#   完整权重 W ∈ R^{out × in}
#   rank i 持有 W[:, i*shard : (i+1)*shard]，shard = in // tp_size
#
# 数据流（承接 ColumnParallelLinear 的输出）：
#   x_i (BATCH, in/tp) —[本地 F.linear]→ partial_y_i (BATCH, out)
#   dist.all_reduce(SUM) → y (BATCH, out)    ← 所有 rank 都有完整结果
#
# 注意：bias 只加在 rank 0，否则会被 all_reduce 放大 tp_size 倍
# ─────────────────────────────────────────────────────────────────────────────

class RowParallelLinear(nn.Module):
    """复刻 nanovllm/layers/linear.py:RowParallelLinear"""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.shard_in = divide(in_features, self.tp_size)
        self.weight = nn.Parameter(torch.empty(out_features, self.shard_in))
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def weight_loader(self, full_weight: torch.Tensor):
        """
        从完整权重按 dim=1（输入维度）切分。
        对应 RowParallelLinear.weight_loader() 中的 narrow(tp_dim=1, ...)
        """
        shard = full_weight.narrow(1, self.tp_rank * self.shard_in, self.shard_in)
        self.weight.data.copy_(shard)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 是 ColumnParallelLinear 的输出，已经按 dim=-1 切好
        bias = self.bias if self.tp_rank == 0 else None   # 只在 rank 0 加 bias！
        partial_y = F.linear(x, self.weight, bias)        # 部分和

        if self.tp_size > 1:
            dist.all_reduce(partial_y, op=dist.ReduceOp.SUM)   # 汇聚为完整输出

        return partial_y


def demo_row_parallel(rank, world_size, device):
    """验证 RowParallelLinear 的 all_reduce 后结果等价于完整计算。"""
    BATCH, IN, OUT = 4, 16, 8

    full_w = broadcast_tensor(torch.randn(OUT, IN, device=device))
    full_x = broadcast_tensor(torch.randn(BATCH, IN, device=device))

    layer = RowParallelLinear(IN, OUT).to(device)
    layer.weight_loader(full_w)

    # 模拟 ColumnParallelLinear 的输出：x 已经按 in-dim 切好
    shard_in = IN // world_size
    x_shard = full_x[:, rank * shard_in : (rank + 1) * shard_in]

    y = layer(x_shard)   # after all_reduce → (BATCH, OUT)

    y_ref = full_x @ full_w.T
    assert torch.allclose(y, y_ref, atol=1e-5)
    log(rank, f"[2] RowParallelLinear      ✓  output shape={list(y.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MergedColumnParallelLinear（gate_up_proj in MLP）
# ─────────────────────────────────────────────────────────────────────────────
# 对应：nanovllm/layers/linear.py  MergedColumnParallelLinear
#
# 用途：Qwen3 的 MLP 中 gate_proj 和 up_proj 被合并为一个大矩阵：
#   [gate_proj; up_proj] ∈ R^{(2*intermediate) × hidden}
#
# TP 切分需要两个 shard 分别独立切：
#   rank i 持有 gate 的第 i 段 + up 的第 i 段
#
# weight_loader 被调用两次（shard_id=0 for gate, shard_id=1 for up），
# 每次把对应片段写入合并权重矩阵的正确偏移位置。
# ─────────────────────────────────────────────────────────────────────────────

class MergedColumnParallelLinear(nn.Module):
    """复刻 nanovllm/layers/linear.py:MergedColumnParallelLinear"""

    def __init__(self, in_features: int, output_sizes: list[int]):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.output_sizes = output_sizes
        total_out = sum(output_sizes)
        self.weight = nn.Parameter(torch.empty(divide(total_out, self.tp_size), in_features))
        nn.init.normal_(self.weight, std=0.02)

    def weight_loader(self, full_weight: torch.Tensor, shard_id: int):
        """
        shard_id=0 → gate_proj，shard_id=1 → up_proj
        每个 shard 按 dim=0 切到本 rank 对应位置。
        对应 MergedColumnParallelLinear.weight_loader()
        """
        shard_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        shard_size   = self.output_sizes[shard_id] // self.tp_size

        # 目标区域：param_data[shard_offset : shard_offset + shard_size, :]
        target = self.weight.data.narrow(0, shard_offset, shard_size)
        # 从完整权重取本 rank 的那一段
        loaded = full_weight.chunk(self.tp_size, 0)[self.tp_rank]
        target.copy_(loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)   # 返回 (BATCH, sum_out/tp)


def demo_merged_column_parallel(rank, world_size, device):
    """验证 MergedColumnParallelLinear 的分片加载和 forward 正确性。"""
    BATCH, IN = 4, 8
    GATE_OUT = UP_OUT = 12   # intermediate_size

    gate_w = broadcast_tensor(torch.randn(GATE_OUT, IN, device=device))
    up_w   = broadcast_tensor(torch.randn(UP_OUT,   IN, device=device))
    x      = broadcast_tensor(torch.randn(BATCH, IN, device=device))

    layer = MergedColumnParallelLinear(IN, [GATE_OUT, UP_OUT]).to(device)
    layer.weight_loader(gate_w, 0)   # gate
    layer.weight_loader(up_w,   1)   # up

    out = layer(x)   # (BATCH, (GATE_OUT+UP_OUT)//tp)

    # 参考：完整矩阵的对应切片
    full_w    = torch.cat([gate_w, up_w], dim=0)   # (2*intermediate, hidden)
    full_out  = x @ full_w.T                        # (BATCH, 2*intermediate)
    shard     = (GATE_OUT + UP_OUT) // world_size
    ref_shard = full_out[:, rank * shard : (rank + 1) * shard]

    assert torch.allclose(out, ref_shard, atol=1e-5)
    log(rank, f"[3] MergedColumnParallel  ✓  output shape={list(out.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. QKVParallelLinear
# ─────────────────────────────────────────────────────────────────────────────
# 对应：nanovllm/layers/linear.py  QKVParallelLinear
#
# 结构：[Q; K; V] 合并为一个投影权重
#   完整权重 ∈ R^{(nQ*hd + nK*hd + nV*hd) × hidden}
#   nQ = total_num_heads,  nK = nV = total_num_kv_heads
#
# TP 切分：按注意力头维度切，每个 rank 持有 nQ/tp 个 Q 头、nK/tp 个 K/V 头
#
# weight_loader 接收 loaded_shard_id ∈ {"q","k","v"}，
# 分三次把从 safetensors 加载的 q_proj / k_proj / v_proj 写入合并权重的正确位置。
# ─────────────────────────────────────────────────────────────────────────────

class QKVParallelLinear(nn.Module):
    """复刻 nanovllm/layers/linear.py:QKVParallelLinear"""

    def __init__(self, hidden: int, head_dim: int, total_q_heads: int, total_kv_heads: int):
        super().__init__()
        self.tp_rank       = dist.get_rank()
        self.tp_size       = dist.get_world_size()
        self.head_dim      = head_dim
        self.num_q_heads   = divide(total_q_heads,  self.tp_size)
        self.num_kv_heads  = divide(total_kv_heads, self.tp_size)
        total_out = (total_q_heads + 2 * total_kv_heads) * head_dim
        self.weight = nn.Parameter(torch.empty(divide(total_out, self.tp_size), hidden))
        nn.init.normal_(self.weight, std=0.02)

    def weight_loader(self, full_weight: torch.Tensor, shard_id: str):
        """
        shard_id ∈ {"q", "k", "v"}
        Q分片大小：num_q_heads * head_dim     写到 offset=0
        K分片大小：num_kv_heads * head_dim    写到 offset=num_q*head_dim
        V分片大小：num_kv_heads * head_dim    写到 offset=num_q*head_dim + num_kv*head_dim
        """
        q_sz = self.num_q_heads  * self.head_dim
        kv_sz= self.num_kv_heads * self.head_dim
        offsets = {"q": 0, "k": q_sz, "v": q_sz + kv_sz}
        sizes   = {"q": q_sz, "k": kv_sz, "v": kv_sz}

        target = self.weight.data.narrow(0, offsets[shard_id], sizes[shard_id])
        loaded = full_weight.chunk(self.tp_size, 0)[self.tp_rank]
        target.copy_(loaded)

    def forward(self, x: torch.Tensor):
        qkv = F.linear(x, self.weight)   # (BATCH, (q+k+v)_per_rank)
        q, k, v = qkv.split([
            self.num_q_heads  * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ], dim=-1)
        return q, k, v


def demo_qkv_parallel(rank, world_size, device):
    """验证 QKVParallelLinear 的分片加载与 forward 结果正确性。"""
    BATCH, H, D = 4, 8, 16   # hidden=8, head_dim=16
    NQ_HEADS = 4              # total Q heads
    NKV_HEADS = 2             # total KV heads (GQA)

    q_w = broadcast_tensor(torch.randn(NQ_HEADS * D, H, device=device))
    k_w = broadcast_tensor(torch.randn(NKV_HEADS * D, H, device=device))
    v_w = broadcast_tensor(torch.randn(NKV_HEADS * D, H, device=device))
    x   = broadcast_tensor(torch.randn(BATCH, H, device=device))

    layer = QKVParallelLinear(H, D, NQ_HEADS, NKV_HEADS).to(device)
    layer.weight_loader(q_w, "q")
    layer.weight_loader(k_w, "k")
    layer.weight_loader(v_w, "v")

    q, k, v = layer(x)

    # 参考
    num_q_per  = NQ_HEADS  // world_size
    num_kv_per = NKV_HEADS // world_size
    q_ref = (x @ q_w.T).view(BATCH, NQ_HEADS,  D)[:, rank*num_q_per  : (rank+1)*num_q_per, :].reshape(BATCH, -1)
    k_ref = (x @ k_w.T).view(BATCH, NKV_HEADS, D)[:, rank*num_kv_per : (rank+1)*num_kv_per, :].reshape(BATCH, -1)
    v_ref = (x @ v_w.T).view(BATCH, NKV_HEADS, D)[:, rank*num_kv_per : (rank+1)*num_kv_per, :].reshape(BATCH, -1)

    assert torch.allclose(q, q_ref, atol=1e-5)
    assert torch.allclose(k, k_ref, atol=1e-5)
    assert torch.allclose(v, v_ref, atol=1e-5)
    log(rank, f"[4] QKVParallelLinear     ✓  q={list(q.shape)}, k={list(k.shape)}, v={list(v.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Attention + MLP 完整 TP 数据流
# ─────────────────────────────────────────────────────────────────────────────
# 展示一个 Transformer 层中 TP 的完整通信拓扑：
#
#  hidden_states (replicated across all ranks)
#       │
#   QKVParallelLinear      ← ColumnParallel，无通信
#       │ (q, k, v slices, local to each rank)
#   local_attention        ← 每个 rank 计算自己的注意力头，无通信
#       │ (o slice)
#   RowParallelLinear      ← all_reduce → 完整 hidden_states (replicated)
#       │
#   MergedColumnParallel   ← gate+up，ColumnParallel，无通信
#       │ (gate, up slices)
#   SiluAndMul             ← 本地激活，无通信
#       │ (activated slice)
#   RowParallelLinear      ← all_reduce → 完整 hidden_states (replicated)
#
# 关键：每个 Transformer 层只有 2 次 all_reduce（attention + MLP 各一次）
# ─────────────────────────────────────────────────────────────────────────────

def demo_transformer_layer_tp(rank, world_size, device):
    """端到端演示一个 TP Transformer 层的数据流，验证与串行版本结果一致。"""
    BATCH, H = 3, 16    # hidden_size=16
    HEAD_DIM = 8
    NQ = NKV = 2        # 每个 total_heads 数，TP 后每 rank 各 NQ/2 个头
    INTERMEDIATE = 32

    # ── 广播所有权重（模拟 safetensors 加载） ──
    q_w   = broadcast_tensor(torch.randn(NQ * HEAD_DIM, H, device=device))
    k_w   = broadcast_tensor(torch.randn(NKV * HEAD_DIM, H, device=device))
    v_w   = broadcast_tensor(torch.randn(NKV * HEAD_DIM, H, device=device))
    o_w   = broadcast_tensor(torch.randn(H, NQ * HEAD_DIM, device=device))  # out_proj
    gate_w= broadcast_tensor(torch.randn(INTERMEDIATE, H, device=device))
    up_w  = broadcast_tensor(torch.randn(INTERMEDIATE, H, device=device))
    down_w= broadcast_tensor(torch.randn(H, INTERMEDIATE, device=device))
    x     = broadcast_tensor(torch.randn(BATCH, H, device=device))

    # ── TP 版本 ──
    # Attention: ColumnParallel QKV
    qkv_layer = QKVParallelLinear(H, HEAD_DIM, NQ, NKV).to(device)
    qkv_layer.weight_loader(q_w, "q") 
    qkv_layer.weight_loader(k_w, "k") 
    qkv_layer.weight_loader(v_w, "v")
    q, k, v = qkv_layer(x)   # local heads only

    # Attention: 本地计算（简化：不含因果掩码/FlashAttn，直接 scaled dot product）
    n_q_local  = NQ  // world_size
    n_kv_local = NKV // world_size
    q_heads = q.view(BATCH, n_q_local, HEAD_DIM)
    k_heads = k.view(BATCH, n_kv_local, HEAD_DIM)
    v_heads = v.view(BATCH, n_kv_local, HEAD_DIM)
    # simplified self-attn: (B, nQ, D) @ (B, D, nKV) → (B, nQ, nKV)
    scale   = HEAD_DIM ** -0.5
    attn    = torch.softmax(q_heads @ k_heads.transpose(-2, -1) * scale, dim=-1)
    o_heads = (attn @ v_heads).reshape(BATCH, -1)   # (BATCH, n_q_local*HEAD_DIM)

    # Attention: RowParallel o_proj
    o_layer = RowParallelLinear(NQ * HEAD_DIM, H).to(device)
    o_layer.weight_loader(o_w)
    attn_out = o_layer(o_heads)   # all_reduce → (BATCH, H)

    # MLP: MergedColumnParallel gate+up
    gate_up_layer = MergedColumnParallelLinear(H, [INTERMEDIATE, INTERMEDIATE]).to(device)
    gate_up_layer.weight_loader(gate_w, 0); gate_up_layer.weight_loader(up_w, 1)
    gate_up = gate_up_layer(attn_out)   # (BATCH, 2*INTERMEDIATE//tp)
    g, u = gate_up.chunk(2, dim=-1)
    activated = F.silu(g) * u   # SiluAndMul

    # MLP: RowParallel down_proj
    down_layer = RowParallelLinear(INTERMEDIATE, H).to(device)
    down_layer.weight_loader(down_w)
    mlp_out = down_layer(activated)   # all_reduce → (BATCH, H)

    # ── 串行参考 ──
    full_qkv_w = torch.cat([q_w, k_w, v_w], dim=0)              # (total_qkv_dim, H)
    qkv_ref    = (x @ full_qkv_w.T)
    q_ref, k_ref, v_ref = qkv_ref.split([NQ*HEAD_DIM, NKV*HEAD_DIM, NKV*HEAD_DIM], dim=-1)
    q_h = q_ref.view(BATCH, NQ, HEAD_DIM)
    k_h = k_ref.view(BATCH, NKV, HEAD_DIM)
    v_h = v_ref.view(BATCH, NKV, HEAD_DIM)
    a   = torch.softmax(q_h @ k_h.transpose(-2, -1) * scale, dim=-1)
    o_ref = (a @ v_h).reshape(BATCH, -1)
    attn_ref = o_ref @ o_w.T

    g_ref = attn_ref @ gate_w.T
    u_ref = attn_ref @ up_w.T
    act_ref = F.silu(g_ref) * u_ref
    mlp_ref = act_ref @ down_w.T

    assert torch.allclose(attn_out, attn_ref, atol=1e-4), "Attention TP 不匹配"
    assert torch.allclose(mlp_out,  mlp_ref,  atol=1e-4), "MLP TP 不匹配"
    log(rank, f"[5] Full Transformer TP   ✓  attn_out={list(attn_out.shape)}, mlp_out={list(mlp_out.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. KV Cache 分配与 TP：num_kv_heads 按 tp_size 切分
# ─────────────────────────────────────────────────────────────────────────────
# 对应：model_runner.py  allocate_kv_cache()
#
#   num_kv_heads = hf_config.num_key_value_heads // self.world_size
#   self.kv_cache = torch.empty(
#       2, num_layers, num_blocks, block_size, num_kv_heads, head_dim
#   )
#
# 意义：每个 rank 只缓存自己负责的 KV heads，内存随 tp_size 线性减少
# ─────────────────────────────────────────────────────────────────────────────

def demo_kvcache_tp(rank, world_size, device):
    total_kv_heads = 8
    num_layers = 4
    block_size = 16
    num_blocks = 32
    head_dim   = 64
    tp_size    = world_size

    num_kv_heads_per_rank = total_kv_heads // tp_size

    # 每个 rank 只分配自己负责的 KV head 数量
    kv_cache = torch.empty(
        2, num_layers, num_blocks, block_size, num_kv_heads_per_rank, head_dim,
        device=device
    )
    total_bytes  = kv_cache.numel() * kv_cache.element_size()
    serial_bytes = total_bytes * tp_size   # 如果不做 TP，单卡需要的总量

    log(rank, (
        f"[6] KV Cache TP  ✓\n"
        f"    total_kv_heads={total_kv_heads}, tp={tp_size}\n"
        f"    per-rank heads={num_kv_heads_per_rank}\n"
        f"    per-rank kv_cache shape={list(kv_cache.shape)}\n"
        f"    per-rank mem={total_bytes/1024/1024:.1f} MB  "
        f"vs serial {serial_bytes/1024/1024:.1f} MB  "
        f"(节省 {(1-1/tp_size)*100:.0f}%)"
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 7. VocabParallelEmbedding + ParallelLMHead（端到端）
# ─────────────────────────────────────────────────────────────────────────────

class VocabParallelEmbedding(nn.Module):
    """复刻 nanovllm/layers/embed_head.py:VocabParallelEmbedding"""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.tp_rank  = dist.get_rank()
        self.tp_size  = dist.get_world_size()
        self.per_rank = divide(vocab_size, self.tp_size)
        self.start    = self.per_rank * self.tp_rank
        self.end      = self.start + self.per_rank
        self.weight   = nn.Parameter(torch.empty(self.per_rank, embed_dim))
        nn.init.normal_(self.weight, std=0.02)

    def weight_loader(self, full_weight: torch.Tensor):
        shard = full_weight.narrow(0, self.tp_rank * self.per_rank, self.per_rank)
        self.weight.data.copy_(shard)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            mask   = (token_ids >= self.start) & (token_ids < self.end)
            safe   = mask.long() * (token_ids - self.start)
            y      = F.embedding(safe, self.weight)
            y      = mask.unsqueeze(-1).float() * y
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        else:
            y = F.embedding(token_ids, self.weight)
        return y


class ParallelLMHead(nn.Module):
    """复刻 nanovllm/layers/embed_head.py:ParallelLMHead"""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.tp_rank  = dist.get_rank()
        self.tp_size  = dist.get_world_size()
        self.per_rank = divide(vocab_size, self.tp_size)
        self.weight   = nn.Parameter(torch.empty(self.per_rank, embed_dim))
        nn.init.normal_(self.weight, std=0.02)

    def weight_loader(self, full_weight: torch.Tensor):
        shard = full_weight.narrow(0, self.tp_rank * self.per_rank, self.per_rank)
        self.weight.data.copy_(shard)

    def forward(self, x: torch.Tensor):
        partial_logits = F.linear(x, self.weight)   # (BATCH, per_rank_vocab)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(partial_logits) for _ in range(self.tp_size)] \
                         if self.tp_rank == 0 else None
            dist.gather(partial_logits, all_logits, dst=0)
            return torch.cat(all_logits, dim=-1) if self.tp_rank == 0 else None
        return partial_logits


def demo_vocab_lmhead(rank, world_size, device):
    VOCAB, EMBED, BATCH = 32, 8, 4

    embed_w = broadcast_tensor(torch.randn(VOCAB, EMBED, device=device))
    lm_w    = broadcast_tensor(torch.randn(VOCAB, EMBED, device=device))
    ids     = broadcast_tensor(torch.randint(0, VOCAB, (BATCH,), device=device))

    emb_layer = VocabParallelEmbedding(VOCAB, EMBED).to(device)
    emb_layer.weight_loader(embed_w)
    emb_out = emb_layer(ids)   # all_reduce → (BATCH, EMBED), replicated

    lm_layer = ParallelLMHead(VOCAB, EMBED).to(device)
    lm_layer.weight_loader(lm_w)
    logits = lm_layer(emb_out)  # gather → (BATCH, VOCAB) only on rank 0

    if rank == 0:
        emb_ref    = F.embedding(ids, embed_w)
        logits_ref = F.linear(emb_out, lm_w)  # using replicated emb_out
        assert torch.allclose(emb_out, emb_ref, atol=1e-5), "VocabEmb 不匹配"
        assert torch.allclose(logits, logits_ref, atol=1e-5), "LMHead 不匹配"
        log(rank, f"[7] VocabEmb+LMHead       ✓  emb={list(emb_out.shape)}, logits={list(logits.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def worker(rank, world_size):
    backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= world_size else "gloo"
    device  = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    setup(rank, world_size, backend)

    log(rank, f"\n{'='*60}")
    log(rank, f" NanoVLLM TP 模型层 Demo  (world_size={world_size}, backend={backend})")
    log(rank, f"{'='*60}")

    demo_column_parallel(rank, world_size, device)
    demo_row_parallel(rank, world_size, device)
    demo_merged_column_parallel(rank, world_size, device)
    demo_qkv_parallel(rank, world_size, device)
    demo_transformer_layer_tp(rank, world_size, device)
    demo_kvcache_tp(rank, world_size, device)
    demo_vocab_lmhead(rank, world_size, device)

    log(rank, f"\n{'='*60}")
    log(rank, " 所有 Demo 通过 ✓")
    log(rank, f"{'='*60}")

    cleanup()


if __name__ == "__main__":
    WORLD_SIZE = 2
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
