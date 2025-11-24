# Deep Dive into Nano-VLLM
## 整体架构  
```shell
                          用户接口层
                        ┌─────────┐
                        │   LLM   │
                        └────┬────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   LLMEngine     │  ← 总指挥官
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ Scheduler   │ │  ← 调度器
                    │ │ BlockManager│ │  ← 内存管家  
                    │ │ ModelRunner │ │  ← 执行引擎
                    │ └─────────────┘ │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model + Layers  │
                    │                 │
                    │ Qwen3Model      │  ← 模型主体
                    │ Attention       │  ← 注意力层
                    │ Sampler         │  ← 采样器
                    └─────────────────┘
```
### 项目结构  
```shell
nanovllm/
├── engine/                    #   推理引擎核心
│   ├── llm_engine.py         #   └── 总协调器，驱动整个推理流程
│   ├── scheduler.py          #   └── 智能调度器，决定执行顺序
│   ├── block_manager.py      #   └── KV缓存内存管理 (PagedAttention核心)
│   ├── model_runner.py       #   └── 单GPU上的模型执行器
│   └── sequence.py           #   └── 请求序列的数据结构
├── layers/                    # ⚙️ 神经网络层实现
│   ├── attention.py          #   └── FlashAttention + KV缓存管理
│   ├── sampler.py            #   └── 从logits采样生成token
│   ├── linear.py             #   └── 支持张量并行的线性层
│   ├── layernorm.py          #   └── RMS LayerNorm
│   ├── rotary_embedding.py   #   └── 旋转位置编码 (RoPE)
│   ├── activation.py         #   └── 激活函数 (SiLU)
│   └── embed_head.py         #   └── 词嵌入和语言模型头
├── models/                    #  ️ 具体模型架构
│   └── qwen3.py              #   └── Qwen3模型完整实现
├── utils/                     #   工具模块
│   ├── context.py            #   └── 全局上下文状态管理
│   └── loader.py             #   └── 模型权重加载器
├── config.py                 # ⚙️ 配置管理
├── llm.py                   #   用户接口入口
└── sampling_params.py       #   采样参数定义
```

## 参考资料  
1. [一条prompt的推理之路](https://www.zhihu.com/search?type=content&q=nanovllm)
2. [nano-vllm源码详细阅读](https://kinnari-blog.vercel.app/posts/nano-vllm/note/)
3. [nano-vllm技术概览](https://zhuanlan.zhihu.com/p/1925484783229698084)