import os
import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.utils.nvtx import nvtx_range


def build_prompts(tokenizer, prompts):
	return [
		tokenizer.apply_chat_template(
			[{"role": "user", "content": prompt}],
			tokenize=False,
			add_generation_prompt=True,
		)
		for prompt in prompts
	]

# Non-default profiler schedule allows user to turn profiler on and off
# on different iteration	s of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(os.path.join("/home/douliyang/large/mlsys/nano-vllm/traces/", f"test_trace_{prof.step_num}.json"))

def profile_run(llm, prompts, sampling_params, trace_dir, tag):
	os.makedirs(trace_dir, exist_ok=True)

	# currently can't enable record_shapes & with_stack due to OOM issue, need to investigate more
	# refer to https://perfetto.dev/docs/visualization/large-traces
	prof = profile(
		activities=[
			ProfilerActivity.CPU, 
			ProfilerActivity.CUDA],
		schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
		on_trace_ready=trace_handler,
		# record_shapes=True,
		# profile_memory=True,
		with_stack=True,
	)
	with prof:
		for iter in range(3):
			with nvtx_range(f"{tag}_iter{iter}"):
				_ = llm.generate(prompts, sampling_params, use_tqdm=False)

			# send a signal to the profiler that the next iteration has started
			prof.step()
			torch.cuda.synchronize()

def main():
	os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

	path = os.path.expanduser("/home/douliyang/large/mlsys/nano-vllm/models/Qwen3-0.6B")
	tokenizer = AutoTokenizer.from_pretrained(path)

	# TP=1 for simplicity, graph enabled to mirror your setup
	llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

	# Warmup: avoid first-run init/graph capture noise
	llm.generate(["warmup"], SamplingParams(temperature=0.6, max_tokens=256), use_tqdm=False)

	long_prompts = build_prompts(
		tokenizer,
		[
			"introduce yourself",
			"list all prime numbers within 100",
			"explain AI compiler stack to me",
			"Hi",
			"give a detailed comparison between vllm, nano-vllm, sglang, mini-sglang, tensorrt, tensorrt-llm",
		],
	)

	# Prefill-heavy: long prompts, minimal decode
	profile_run(
		llm,
		long_prompts,
		SamplingParams(temperature=0.6, max_tokens=256),
		"./traces/prefill",
		"prefill_heavy",
	)

if __name__ == "__main__":
	main()
