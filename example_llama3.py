import os
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    # 确保路径正确
    path = os.path.expanduser("/home/douliyang/large/mlsys/nano-vllm/models/Llama3-8B")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

    # --- 修复代码：手动补全 chat_template ---
    # 如果 tokenizer 没有内置模板，手动设置 Llama3 标准模板
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        )
    # --- 修复结束 ---

    # 2. 初始化 LLM
    # tensor_parallel_size=2 开启 TP
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # 3. 准备 Prompt
    user_prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
        "explain AI compiler stack to me",
        "Hi",
        "give a detailed comparison between vllm, nano-vllm, sglang, mini-sglang, tensorrt, tensorrt-llm",
    ]
    
    # 使用 apply_chat_template 将对话转换为模型格式
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in user_prompts
    ]

    # 4. 生成
    outputs = llm.generate(prompts, sampling_params)

    # 5. 打印结果
    for prompt, output in zip(user_prompts, outputs):
        print("\n" + "="*50)
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")

if __name__ == "__main__":
    main()