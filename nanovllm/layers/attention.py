import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# key tensor (dense, 当前 forward 输出)
# [N, num_heads, head_dim]  (stride: [D, head_dim, 1])

# slot_mapping
# [N]  →  [slot_0, slot_1, ..., slot_N-1]
#          = block_table[block_i] * block_size + offset

#                     Triton kernel (N个并行program)
#                    ┌─────────────────────────────┐
#   program idx=0:   │ key[0] ──→ k_cache[slot_0] │
#   program idx=1:   │ key[1] ──→ k_cache[slot_1] │
#   program idx=i:   │ key[i] ──→ k_cache[slot_i] │
#                    └─────────────────────────────┘

# k_cache (物理 cache，预分配)
# [num_blocks, block_size, num_kv_heads, head_dim]
#  等价视图：[num_slots, D]
#   slot_i 行 = block_table[i//block_size] 对应块的第 (i%block_size) 个位置


# triton kernel，完成kv cache的写入  
# slot_mapping 告诉我们每个 token 的 KV 应该写到物理缓存的哪个位置

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return     # sequence padding相关
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # slot是槽位，是blockid * BLOCK_SIZE + block_offset
    # D是num_kv_heads, head_dim两个维度展开的一维
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape   # N = 本次 forward 的 token 数
    D = num_heads * head_dim

    # 检查内存是连续排列的（triton 直接操作指针，必须保证 stride 正确）
    assert key.stride(-1) == 1           # head_dim 维度是连续的
    assert key.stride(1) == head_dim     # 相邻 head 之间相差 head_dim 个元素
    assert k_cache.stride(1) == D        # cache 的 block_offset 维步长 = D

    assert slot_mapping.numel() == N     # 每个 token 对应一个 slot

    # 启动 N 个 triton program，每个处理 1 个 token
    store_kvcache_kernel[(N,)](
        key, key.stride(0),   # key[token] 到 key[token+1] 跨 stride(0) 个元素
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping,
        D                     # constexpr，编译期固定，用于 tl.arange
    )


# 借助于flashattention优化，实现注意力机制  
# 1. 支持flash attention
# 2. 支持kv cache
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([]) # numel() == 0


    # flash attention接口解读：
    # 1. prefill 接口
    # o = flash_attn_varlen_func(
    #     q,                              # [total_q, num_heads, head_dim]
    #     k,                              # [total_k, num_kv_heads, head_dim]
    #     v,                              # [total_k, num_kv_heads, head_dim]
    #     cu_seqlens_q=cu_seqlens_q,      # [batch_size + 1]
    #     cu_seqlens_k=cu_seqlens_k,      # [batch_size + 1]
    #     max_seqlen_q=max_seqlen_q,      # int
    #     max_seqlen_k=max_seqlen_k,      # int
    #     softmax_scale=scale,
    #     causal=True,
    #     block_table=block_tables        # [batch_size, max_blocks] 或 None
    # )

    # 2. decode接口
    # o = flash_attn_with_kvcache(
    #     q.unsqueeze(1),                 # [batch_size, 1, num_heads, head_dim]
    #     k_cache,                        # [num_blocks, block_size, num_kv_heads, head_dim]
    #     v_cache,                        # [num_blocks, block_size, num_kv_heads, head_dim]
    #     cache_seqlens=context_lens,     # [batch_size]
    #     block_table=block_tables,       # [batch_size, max_blocks]
    #     softmax_scale=scale,
    #     causal=True
    # )

    # 对于flashattn + prefix cache的详细解读：
    # 1. prepare_prefill 检测到 cu_seqlens_k[-1] > cu_seqlens_q[-1]
    #    → 说明部分 token 已缓存
    #    → 构造 block_tables

    # 2. Attention.forward:
    #    a. store_kvcache 写入未缓存部分的 KV（slot_mapping 只包含未缓存的槽位）
    #    b. block_tables 不为 None，所以把 k, v 替换为 k_cache, v_cache
    #    c. flash_attn_varlen_func 通过 block_table 读取分页的完整 KV

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 写入 KV Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # prepare阶段，q只有一个token，添加一个维度变成[Batch，1, num_heads, head_dim]，k和v直接使用cache，flash attention的接口会根据context.slot_mapping自动在cache里找到对应的kv
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
