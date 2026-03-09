import torch
from torch import nn
import torch.distributed as dist

from transformers import LlamaConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Llama3Attention(nn.Module):
    """Multi-head GQA attention for Llama3.

    与 Qwen3Attention 的关键区别：
    - 无 q_norm / k_norm（那是 Qwen3 特有的，用于稳定大规模训练）
    - qkv_bias 固定 False
    - rope_theta 默认 500000（Qwen3 是 1000000）
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int | None,
        rope_theta: float,
        rope_scaling: dict | None,
    ) -> None:
        # TP 并行首先按 head 维度切割
        super().__init__()
        # 考虑 GQA 场景，num_kv_heads 可能不等于 num_heads
        self.total_num_heads = num_heads
        self.tp = dist.get_world_size()
        assert self.total_num_heads % self.tp == 0

        self.num_heads = self.total_num_heads // self.tp
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % self.tp == 0
        self.num_kv_heads = self.total_num_kv_heads // self.tp

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,  # Llama3 固定无 bias
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 旋转编码；
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=None,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 注意：Llama3 不需要 q_norm / k_norm，此处不加

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        query, key, val = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # reshape 为 (seq_len, num_heads, head_dim) 供 flash_attn 使用
        query = query.view(-1, self.num_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        val = val.view(-1, self.num_kv_heads, self.head_dim)

        # RoPE；Llama3 直接旋转，无 q/k norm
        q, k = self.rotary_emb(positions, query, key)
        attn_output = self.attn(q, k, val)
        # flatten(1, -1): 只合并 head 维度，保留 seq_len 维度
        return self.o_proj(attn_output.flatten(1, -1))


class Llama3MLP(nn.Module):
    """SwiGLU FFN，与 Qwen3MLP 结构完全一致。

    gate_up_proj 把 gate 和 up 两路合并成一个矩阵乘，
    SiluAndMul 做切分 + silu(gate)*up，再经 down_proj。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # 将两次线性变换合并成一个 MergedColumnParallelLinear，输出维度 intermediate_size*2
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=bias,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)


class Llama3DecoderLayer(nn.Module):
    """单个 Transformer Block。

    Pre-Norm + fused add-norm 结构，属性名与 HF checkpoint 保持一致，
    保证 load_model() 能按名字索引并正确加载权重。
    """

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.self_attn = Llama3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            # Llama3 rope_theta 默认 500000；Llama3.1 通常会在 config 里显式指定
            rope_theta=getattr(config, "rope_theta", 500000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Llama3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            bias=getattr(config, "mlp_bias", False),
        )
        # 属性名必须与 HF 权重名一致，否则 load_model 找不到对应参数
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # fused add-norm：把 residual 加法与 RMSNorm 合并，减少一次 HBM 读写
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Llama3Model(nn.Module):
    """Llama3 Transformer 主体（不含 lm_head）。

    layer_type 参数预留给 EAGLE 等投机采样场景，
    允许替换为自定义子类而不改动主流程。
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_type: type[nn.Module] = Llama3DecoderLayer,
    ) -> None:
        super().__init__()
        # 属性名 embed_tokens 与 HF 权重 model.embed_tokens.weight 对应
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # TODO(leon): 修改支持 PP 并行时，layers 按 stage 切分
        self.layers = nn.ModuleList([layer_type(config) for _ in range(config.num_hidden_layers)])
        # 属性名 norm 与 HF 权重 model.norm.weight 对应
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层结束后把 residual 加回再 norm（fused add-norm 的收尾）
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# 自回归 Llama3 模型的入口（后续可以扩展诸如 EAGLE 等优化）
class Llama3ForCausalLM(nn.Module):
    # packed_modules_mapping 告诉 load_model() 将 HF 的独立权重合并到本项目的联合参数
    # 例如 q_proj/k_proj/v_proj -> 一张 qkv_proj 矩阵（减少三次 matmul 为一次）
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        layer_type: type[nn.Module] = Llama3DecoderLayer,
    ) -> None:
        super().__init__()
        self.model = Llama3Model(config=config, layer_type=layer_type)
        # lm_head 必须在 tie_word_embeddings 赋值前初始化
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, bias=False)
        if config.tie_word_embeddings:
            # 小模型共享 embedding 权重，节省显存
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # ParallelLMHead.forward 在 prefill 时自动只取每条序列最后一个 token 的 hidden state
        return self.lm_head(hidden_states)
