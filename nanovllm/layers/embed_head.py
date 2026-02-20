import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# TP并行版本的embedding和lm_head，
# 核心思想是把词表切分成tp_size份，
# 每个rank只负责自己那份的embedding和lm_head计算，最后通过通信合并结果
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0

        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 只处理属于当前 rank 的词表范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        # x维度：[B,S]
        # weight维度：[V,H]
        # F.embedding作用是把每个 token id 当成行索引，
        # 去 weight 里取那一整行向量。
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 不在范围内的位置置零
            # mask是[B,S],需要扩展到[B,S,1]才能和y相乘，y是[B,S,H]
            # 然后 all_reduce 聚合
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


# ParallelLMHead输入是 hidden_states，输出是 logits
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # Prefill 时只取每个序列的最后一个位置
            # cu_seqlens_q 是 累积长度数组
            # eg. 3 个序列，长度分别 5, 7, 4
            # cu_seqlens_q = [0, 5, 12, 16]
            # last_indices = [4, 11, 15]，正好是每个序列的最后一个 token 的位置索引
            last_indices = context.cu_seqlens_q[1:] - 1
            # 注意，x在decode阶段做过flatten，所以维度是(total_tokens, H)
            x = x[last_indices].contiguous()  # 取每个序列的最后一个位置，并用新的tensor保持内存连续性
        
        # x [tokens,H]，weight是[V,H]，[total_tokens,V]
        # 把 hidden 向量投影到词表空间
        # 计算它和每个词的相似度
        # 得到预测概率的原始分数（logits）
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 每个rank覆盖部分logits，利用dist库gather到rank0
            dist.gather(logits,           # 当前rank数据
                        all_logits,       # 目标rank 用来存所有数据的 list
                        0)                # 目标rank，表示gather 到rank0
            # 按照最后一个维度做拼接
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None  
        return logits
