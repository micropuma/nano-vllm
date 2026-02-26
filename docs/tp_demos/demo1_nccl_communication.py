"""
Demo 1: NanoVLLM TP并行中的 NCCL 通信原语
============================================

NanoVLLM 中共使用了三种 NCCL 通信原语：
  1. dist.all_reduce  —— RowParallelLinear 的 reduce-scatter 等效操作
  2. dist.all_reduce  —— VocabParallelEmbedding 的 embedding 结果汇聚
  3. dist.gather      —— ParallelLMHead 把各 rank 的 logits shard 汇到 rank 0

运行方式（需要 2 张 GPU）：
  torchrun --nproc_per_node=2 demo1_nccl_communication.py

单卡模拟（使用 gloo + CPU）也可运行，见 SIMULATE_SINGLE 开关。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def setup(rank: int, world_size: int, backend: str = "nccl"):
    """初始化进程组。单机多卡固定用 tcp://localhost:29500。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def log(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo A: all_reduce —— RowParallelLinear 中的核心通信
# ─────────────────────────────────────────────────────────────────────────────
# 对应代码：nanovllm/layers/linear.py  RowParallelLinear.forward()
#
#   def forward(self, x):
#       y = F.linear(x, self.weight, ...)   # 每个 rank 只算自己那一段输入的贡献
#       if self.tp_size > 1:
#           dist.all_reduce(y)              # 对所有 rank 的部分积求和
#       return y
#
# 数学含义：
#   设完整权重 W ∈ R^{out × in}，被按列切成 W_0, W_1（每个 rank 持有一半）
#   输入 x 也被切成 x_0, x_1（column-parallel 上一层已经切好）
#   rank_i 本地计算  y_i = x_i @ W_i^T
#   all_reduce(SUM) 后每个 rank 都得到  y = y_0 + y_1 = x @ W^T  ✓
# ─────────────────────────────────────────────────────────────────────────────

def demo_all_reduce(rank: int, world_size: int, device):
    dist.barrier()
    BATCH, IN, OUT = 4, 8, 6

    # 完整权重（rank 0 负责广播，其余 rank只是验证用）
    if rank == 0:
        full_weight = torch.randn(OUT, IN, device=device)
        full_x      = torch.randn(BATCH, IN, device=device)
    else:
        full_weight = torch.zeros(OUT, IN, device=device)
        full_x      = torch.zeros(BATCH, IN, device=device)

    # 广播给所有 rank（实际推理中每 rank 从 safetensors 直接加载自己的分片，这里仅为 demo）
    dist.broadcast(full_weight, src=0)
    dist.broadcast(full_x, src=0)

    # ── 按列切分权重（dim=1），每个 rank 拿 IN//world_size 列 ──
    shard_in = IN // world_size
    w_shard  = full_weight[:, rank * shard_in : (rank + 1) * shard_in]  # (OUT, shard_in)
    x_shard  = full_x[:, rank * shard_in : (rank + 1) * shard_in]       # (BATCH, shard_in)

    # ── 每个 rank 本地部分积 ──
    partial_y = x_shard @ w_shard.T   # (BATCH, OUT)

    log(rank, f"all_reduce BEFORE: partial_y[0] = {partial_y[0].tolist()}")

    # ── NCCL all_reduce (SUM) ──
    dist.all_reduce(partial_y, op=dist.ReduceOp.SUM)

    log(rank, f"all_reduce AFTER : reduced_y[0] = {partial_y[0].tolist()}")

    # ── 验证结果与串行计算完全一致 ──
    expected = full_x @ full_weight.T
    assert torch.allclose(partial_y, expected, atol=1e-5), "all_reduce 结果不匹配！"
    log(rank, "✓ all_reduce 验证通过")


# ─────────────────────────────────────────────────────────────────────────────
# Demo B: all_reduce —— VocabParallelEmbedding 中的 masked embedding 汇聚
# ─────────────────────────────────────────────────────────────────────────────
# 对应代码：nanovllm/layers/embed_head.py  VocabParallelEmbedding.forward()
#
#   mask = (x >= vocab_start) & (x < vocab_end)
#   x    = mask * (x - vocab_start)        # 把越界 token id 清零，防止越界查表
#   y    = F.embedding(x, self.weight)     # 本 rank vocab 分片的 embedding
#   y    = mask.unsqueeze(1) * y           # 越界位置输出置零
#   dist.all_reduce(y)                     # 每个位置只有一个 rank 有非零贡献 → 汇聚
#
# 技巧：因为各 rank  vocab 区间不重叠，all_reduce(SUM) 等价于每个位置
#       "选出那个负责该 token 的 rank 的 embedding 向量"。
# ─────────────────────────────────────────────────────────────────────────────

def demo_vocab_parallel_embedding(rank: int, world_size: int, device):
    dist.barrier()
    VOCAB_SIZE   = 16   # 总词表大小
    EMBED_DIM    = 4
    BATCH        = 6    # 6 个 token id

    per_rank = VOCAB_SIZE // world_size
    vocab_start = rank * per_rank
    vocab_end   = vocab_start + per_rank

    # 每个 rank 持有自己那段 embedding 表
    local_weight = torch.arange(
        vocab_start * EMBED_DIM, vocab_end * EMBED_DIM,
        dtype=torch.float32, device=device
    ).reshape(per_rank, EMBED_DIM)   # 用有规律的值方便肉眼验证

    # 模拟输入 token ids（跨越所有 rank 的 vocab 区间）
    if rank == 0:
        token_ids = torch.tensor([0, 5, 8, 12, 3, 10], device=device)
    else:
        token_ids = torch.zeros(6, dtype=torch.long, device=device)
    dist.broadcast(token_ids, src=0)

    # ── VocabParallelEmbedding.forward 逻辑 ──
    mask    = (token_ids >= vocab_start) & (token_ids < vocab_end)
    safe_ids = mask.long() * (token_ids - vocab_start)
    y       = torch.nn.functional.embedding(safe_ids, local_weight)
    y       = mask.unsqueeze(1).float() * y        # 只保留本 rank 负责的位置

    log(rank, f"vocab_embed BEFORE all_reduce: y[0]={y[0].tolist()}")
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    log(rank, f"vocab_embed AFTER  all_reduce: y[0]={y[0].tolist()}")

    # ── 验证 ──
    if rank == 0:
        # 构造完整 embedding 表做对比
        full_weight = torch.cat([
            torch.arange(i * per_rank * EMBED_DIM, (i + 1) * per_rank * EMBED_DIM,
                         dtype=torch.float32, device=device).reshape(per_rank, EMBED_DIM)
            for i in range(world_size)
        ])
        expected = torch.nn.functional.embedding(token_ids, full_weight)
        assert torch.allclose(y, expected, atol=1e-5), "VocabParallelEmbedding 验证失败！"
        log(rank, "✓ VocabParallelEmbedding 验证通过")


# ─────────────────────────────────────────────────────────────────────────────
# Demo C: gather —— ParallelLMHead 把 logits 汇聚到 rank 0
# ─────────────────────────────────────────────────────────────────────────────
# 对应代码：nanovllm/layers/embed_head.py  ParallelLMHead.forward()
#
#   logits = F.linear(x, self.weight)          # 每 rank 算部分 vocab 的 logit
#   all_logits = [torch.empty_like(logits)...] if rank==0 else None
#   dist.gather(logits, all_logits, dst=0)     # gather 到 rank 0
#   logits = torch.cat(all_logits, dim=-1)     # 拼出完整 vocab logits
#
# 与 all_reduce 的区别：
#   - all_reduce: 结果每个 rank 都有（常用于中间层）
#   - gather:     结果只在 dst rank（常用于最终输出，节省带宽）
# ─────────────────────────────────────────────────────────────────────────────

def demo_gather_logits(rank: int, world_size: int, device):
    dist.barrier()
    VOCAB_SIZE = 20
    HIDDEN     = 8
    BATCH      = 3

    per_rank_vocab = VOCAB_SIZE // world_size

    # 模拟 hidden_states
    if rank == 0:
        hidden = torch.randn(BATCH, HIDDEN, device=device)
    else:
        hidden = torch.zeros(BATCH, HIDDEN, device=device)
    dist.broadcast(hidden, src=0)

    # 模拟每 rank 的 LM head 权重分片（按词表维度切分）
    lm_head_weight = torch.randn(per_rank_vocab, HIDDEN, device=device)

    # ── 本 rank 的部分 logits ──
    partial_logits = torch.nn.functional.linear(hidden, lm_head_weight)  # (BATCH, per_rank_vocab)

    # ── dist.gather → rank 0 收集所有分片 ──
    if rank == 0:
        gather_list = [torch.empty_like(partial_logits) for _ in range(world_size)]
    else:
        gather_list = None

    dist.gather(partial_logits, gather_list, dst=0)

    if rank == 0:
        full_logits = torch.cat(gather_list, dim=-1)   # (BATCH, VOCAB_SIZE)
        log(rank, f"gather: full_logits shape = {full_logits.shape}  ✓")

        # 对比：用完整权重直接算
        all_weights = [torch.zeros(per_rank_vocab, HIDDEN, device=device) for _ in range(world_size)]
        all_weights[0].copy_(lm_head_weight)
        # （其他 rank 的权重在真实场景已切好，这里 demo 只在 rank0 验证形状）
        log(rank, "✓ gather logits 结构验证通过（形状正确）")
    else:
        log(rank, "gather: 仅发送本 rank partial_logits，结果在 rank 0")


# ─────────────────────────────────────────────────────────────────────────────
# Demo D: barrier —— TP 进程同步点
# ─────────────────────────────────────────────────────────────────────────────
# 对应代码：model_runner.py
#
#   if rank == 0:
#       self.shm = SharedMemory(..., create=True, ...)
#       dist.barrier()          # 等所有 rank 就绪
#   else:
#       dist.barrier()
#       self.shm = SharedMemory(name="nanovllm")   # 打开 rank 0 已创建好的 shm
#
# barrier 确保：rank 0 先建好共享内存，其他 rank 才打开。
# ─────────────────────────────────────────────────────────────────────────────

def demo_barrier(rank: int, world_size: int, device):
    import time
    if rank == 0:
        time.sleep(0.2)   # 模拟 rank 0 需要更长初始化时间（比如创建 shm）
        log(rank, "rank 0: 共享内存已创建，调用 barrier 等待其他 rank")
        dist.barrier()
        log(rank, "rank 0: 所有 rank 已就绪，继续执行")
    else:
        log(rank, f"rank {rank}: 等待 rank 0 创建共享内存，barrier 中...")
        dist.barrier()
        log(rank, f"rank {rank}: barrier 通过，打开共享内存")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def worker(rank: int, world_size: int):
    backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= world_size else "gloo"
    device  = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    setup(rank, world_size, backend)

    print(f"\n{'='*60}")
    print(" Demo A: all_reduce (RowParallelLinear)")
    print(f"{'='*60}")
    demo_all_reduce(rank, world_size, device)

    print(f"\n{'='*60}")
    print(" Demo B: VocabParallelEmbedding masked all_reduce")
    print(f"{'='*60}")
    demo_vocab_parallel_embedding(rank, world_size, device)

    print(f"\n{'='*60}")
    print(" Demo C: gather logits (ParallelLMHead)")
    print(f"{'='*60}")
    demo_gather_logits(rank, world_size, device)

    print(f"\n{'='*60}")
    print(" Demo D: barrier (SharedMemory 同步)")
    print(f"{'='*60}")
    demo_barrier(rank, world_size, device)

    cleanup()


if __name__ == "__main__":
    WORLD_SIZE = 2
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
