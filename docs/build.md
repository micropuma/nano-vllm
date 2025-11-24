# 源码构建  
## 构建步骤
1. 从github clone源代码  
    ```shell
    git clone https://github.com/micropuma/nano-vllm.git
    ```  
2. 启动conda环境  
    ```shell
    conda create -n nanovllm python=3.10
    conda activate nanovllm
    ```
3. 针对自己的cuda版本，修改pyproject.toml配置文件，来安装指定版本的torch等包  
    ```shell
    nvcc --version
    ```
    我的cuda版本是12.0，根据pyproject.toml配置中，torch版本要高于2.4，在pytorch官方找到如下wheel包：https://download.pytorch.org/whl/cu121/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl  
    设置pyproject.toml的dependencies字段为：  
    ```shell
    dependencies = [
        "torch @ https://download.pytorch.org/whl/cu121/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl",
        "triton>=3.0.0",
        "transformers>=4.51.0",
        "xxhash",
    ]
    ```  
    注意，这里删除了flash-attn，因为面向torch2.4.1+cu121，flashattention没有官方wheel包。需要源码编译。  
4. 源码构建FlashAttention  
    * clone 源码  
        ```shell
        git clone https://github.com/Dao-AILab/flash-attention
        ```
    * checkout 到v2.6.3版本。该版本提供torch(2.0 - 2.4)+cu(118/123)+cp(38-311)，我们的版本在这个区间内。    
        ```shell
        cd flash-attention
        git checkout v2.6.3
        ```
    * 针对setup.py做如下修改：  
        ```shell
        # before 
        os.rename(wheel_filename, wheel_path)
        # after
        shutil.move(wheel_filename, wheel_path)
        ```  
        如果不做修改，后续会报错：  
        ```shell
        error: [Errno 18] Invalid cross-device link:
        ```
        具体原理参考[StackOverflow QA](https://stackoverflow.com/questions/42392600/oserror-errno-18-invalid-cross-device-link)。  

    * 源码构建  
        ```shell
        MAX_JOBS=1 pip install . --no-build-isolation
        ```  
        FlashAttention默认使用64线程，需要96G内存。资源不够的情况下使用`MAX_JOBS`降低线程数。`--no-build-isolation`取消使用独立build-system来构建，使得构建过程可以使用conda环境。   
5. 源码构建nano-vllm
    ```shell
    pip install -e . --no-build-isolation
    ```
6. 验证构建是否成功  
    ```python
    from nanovllm import LLM, SamplingParams
    import torch
    import flash_attn

    print("Nano-vLLM imported successfully")
    print("PyTorch version:", torch.__version__, "CUDA version:", torch.version.cuda)
    print("Flash-Attn version:", flash_attn.__version__)
    ```

## Quick Start  
1. 模型准备  
    首先需要安装huggingface-hub来拉取模型  
    ```shell
    pip install "huggingface-hub<=1.0"
    ```
    使用如下命令拉去模型：  
    ```shell
    huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
        --local-dir ~/huggingface/Qwen3-0.6B/ \
        --local-dir-use-symlinks False
    ```
2. 使用bench.py脚本做benchmark
    ```shell
    python bench.py
    ```
    在我的RTX3090结果如下：  
    ```shell
    Generating: 100%|█████████████████████████████████████████████████████████████| 1/1 [00:26<00:00, 26.99s/it, Prefill=0tok/s, Decode=328tok/s]
    Total: 133966tok, Time: 48.24s, Throughput: 2777.02tok/s
    ```
