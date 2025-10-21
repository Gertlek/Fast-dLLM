import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y"}


@register_model("qwen25_flash")
class Qwen25FlashEvalHarness(LM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype_str: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        max_new_tokens: int = 512,
        batch_size: int = 1,
        temperature: float = 0.0,
        top_p: float = 0.95,
        show_progress: bool = True,
        show_speed: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.attn_implementation = attn_implementation
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.show_progress = _to_bool(show_progress)
        self.show_speed = _to_bool(show_speed)
        self._world_size = 1
        self._rank = 0

        # Timing statistics
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_prefill_tokens = 0
        self.total_decode_tokens = 0

        # Determine dtype
        if dtype_str == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_str == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Load model with specified attention implementation
        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.model_name

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
        """Apply chat template to format messages."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests) -> List[str]:
        """Generate with batching support and prefill/decode timing."""
        output = [None] * len(requests)
        
        progress_bar = None
        if self.rank == 0 and self.show_progress:
            progress_bar = tqdm(total=len(requests), desc="Generating", leave=False)

        with torch.inference_mode():
            # Process in batches
            for batch_start in range(0, len(requests), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(requests))
                batch_requests = requests[batch_start:batch_end]
                
                # Prepare batch
                prompts = [req.args[0] for req in batch_requests]
                
                # Tokenize with padding
                model_inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to(self.device)
                
                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]
                input_lengths = attention_mask.sum(dim=1).tolist()
                
                # Measure prefill time (first forward pass)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                prefill_start = time.time()
                
                # Do a single forward pass to measure prefill
                with torch.no_grad():
                    _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                prefill_end = time.time()
                prefill_time = prefill_end - prefill_start
                
                # Measure decode time (full generation)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                decode_start = time.time()
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                decode_end = time.time()
                decode_time = decode_end - decode_start
                
                # Update statistics
                batch_prefill_tokens = sum(input_lengths)
                self.total_prefill_tokens += batch_prefill_tokens
                self.total_prefill_time += prefill_time
                
                # Decode each sample in batch
                for batch_idx, (input_len, gen_ids) in enumerate(zip(input_lengths, generated_ids)):
                    continuation = gen_ids[input_len:]
                    num_decode_tokens = continuation.numel()
                    self.total_decode_tokens += num_decode_tokens
                    
                    decoded = self.tokenizer.decode(continuation, skip_special_tokens=True)
                    output[batch_start + batch_idx] = decoded.strip()
                
                # Subtract prefill from total decode time
                actual_decode_time = decode_time - prefill_time
                self.total_decode_time += actual_decode_time
                
                if progress_bar is not None:
                    progress_bar.update(len(batch_requests))

        if progress_bar is not None:
            progress_bar.close()

        if self.show_speed and self.rank == 0:
            total_time = self.total_prefill_time + self.total_decode_time
            if total_time > 0:
                print(f"\n{'=' * 80}")
                print(f"Throughput Metrics ({self.attn_implementation or 'default'})")
                print(f"{'=' * 80}")
                print(f"Prefill:")
                print(f"  Total tokens: {self.total_prefill_tokens}")
                print(f"  Total time: {self.total_prefill_time:.2f}s")
                print(f"  Throughput: {self.total_prefill_tokens / self.total_prefill_time:.2f} tokens/s")
                print(f"  Avg tokens per request: {self.total_prefill_tokens / len(requests):.1f}")
                print(f"\nDecode:")
                print(f"  Total tokens: {self.total_decode_tokens}")
                print(f"  Total time: {self.total_decode_time:.2f}s")
                print(f"  Throughput: {self.total_decode_tokens / self.total_decode_time:.2f} tokens/s")
                print(f"  Avg tokens per request: {self.total_decode_tokens / len(requests):.1f}")
                print(f"\nOverall:")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  Total tokens: {self.total_prefill_tokens + self.total_decode_tokens}")
                print(f"  Throughput: {(self.total_prefill_tokens + self.total_decode_tokens) / total_time:.2f} tokens/s")
                print(f"  Avg latency per request: {total_time / len(requests):.3f}s")
                print(f"  Batch size: {self.batch_size}")
                print(f"{'=' * 80}\n")

        return output


def _build_cli_args(namespace, use_flash: bool) -> List[str]:
    attn_impl = "flash_attention_2" if use_flash else namespace.non_flash_attn_impl
    attn_label = "flash" if use_flash else namespace.non_flash_attn_impl

    model_args = [
        f"model_name={namespace.model_name}",
        f"attn_implementation={attn_impl}",
        f"dtype_str={namespace.dtype}",
        f"max_new_tokens={namespace.max_new_tokens}",
        f"temperature={namespace.temperature}",
        f"top_p={namespace.top_p}",
        "show_progress=True",
        "show_speed=True",
    ]

    output_path = Path(namespace.output_dir) / f"gsm8k_qwen25_{attn_label}_bs{namespace.batch_size}.json"

    args = [
        "lm_eval",
        "--model",
        "qwen25_flash",
        "--tasks",
        "gsm8k",
        "--batch_size",
        str(namespace.batch_size),
        "--num_fewshot",
        str(namespace.num_fewshot),
        "--confirm_run_unsafe_code",
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--model_args",
        ",".join(model_args),
        "--output_path",
        str(output_path),
    ]

    if namespace.limit is not None:
        args.extend(["--limit", str(namespace.limit)])

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5 on GSM8K with and without flash attention.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--non_flash_attn_impl",
        type=str,
        default="eager",
        help="Attention backend when flash is disabled (eager, sdpa).",
    )
    parser.add_argument("--methods", choices=["flash", "no_flash", "both"], default="both")
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of examples (None for all).")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    if args.limit is not None and args.limit <= 0:
        args.limit = None

    methods = ["flash", "no_flash"] if args.methods == "both" else [args.methods]

    for method in methods:
        use_flash = method == "flash"
        label = "FLASH" if use_flash else "NO_FLASH"

        if method != methods[0] and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print(f"Evaluating Qwen 2.5 on GSM8K with {label} attention")
        print("=" * 80 + "\n")

        sys.argv = _build_cli_args(args, use_flash)
        cli_evaluate()


if __name__ == "__main__":
    main()
    
    
                                                                                                                                                                                                        
# 025-10-17:09:30:58 INFO     [evaluator:574] Running generate_until requests
                                                                                                                                                                       
# ================================================================================
# Throughput Metrics (flash_attention_2)
# ================================================================================
# Prefill:
#   Total tokens: 108757
#   Total time: 4.98s
#   Throughput: 21834.34 tokens/s
#   Avg tokens per request: 1087.6

# Decode:
#   Total tokens: 20747
#   Total time: 308.93s
#   Throughput: 67.16 tokens/s
#   Avg tokens per request: 207.5

# Overall:
#   Total time: 313.91s
#   Total tokens: 129504
#   Throughput: 412.55 tokens/s
#   Avg latency per request: 3.139s
#   Batch size: 1
# ================================================================================

# 2025-10-17:09:36:20 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
# qwen25_flash (model_name=Qwen/Qwen2.5-7B-Instruct,attn_implementation=flash_attention_2,dtype_str=bfloat16,max_new_tokens=512,temperature=1.0,top_p=1.0,show_progress=True,show_speed=True), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.82|±  |0.0386|
# |     |       |strict-match    |     5|exact_match|↑  | 0.23|±  |0.0423|


# ================================================================================
# Evaluating Qwen 2.5 on GSM8K with NO_FLASH attention
# ================================================================================

# 2025-10-17:09:36:22 WARNING  [__main__:369]  --limit SHOULD ONLY BE USED FOR TESTING.REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.
# 2025-10-17:09:36:22 INFO     [__main__:446] Selected Tasks: ['gsm8k']
# 2025-10-17:09:36:22 INFO     [evaluator:202] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
# 2025-10-17:09:36:22 INFO     [evaluator:240] Initializing qwen25_flash model, with arguments: {'model_name': 'Qwen/Qwen2.5-7B-Instruct', 'attn_implementation': 'eager', 'dtype_str':
#         'bfloat16', 'max_new_tokens': 512, 'temperature': 1.0, 'top_p': 1.0, 'show_progress': True, 'show_speed': True}
# 2025-10-17:09:36:22 INFO     [accelerate.utils.modeling:1004] We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
# Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.79it/s]
# 2025-10-17:09:36:27 INFO     [evaluator:305] gsm8k: Using gen_kwargs: {'until': ['Question:', '</s>', '<|im_end|>'], 'do_sample': False, 'temperature': 0.0}
# 2025-10-17:09:36:27 WARNING  [evaluator:324] Overwriting default num_fewshot of gsm8k from 5 to 5
# 2025-10-17:09:36:27 WARNING  [evaluator:480] Chat template formatting change affects loglikelihood and multiple-choice tasks. See docs/chat-template-readme.md for details.
# 2025-10-17:09:36:27 INFO     [api.task:434] Building contexts for gsm8k on rank 0...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 465.78it/s]
# 2025-10-17:09:36:27 INFO     [evaluator:574] Running generate_until requests
                                                                                                                                                                       
# ================================================================================
# Throughput Metrics (eager)
# ================================================================================
# Prefill:
#   Total tokens: 108757
#   Total time: 6.18s
#   Throughput: 17602.05 tokens/s
#   Avg tokens per request: 1087.6

# Decode:
#   Total tokens: 20954
#   Total time: 356.69s
#   Throughput: 58.74 tokens/s
#   Avg tokens per request: 209.5

# Overall:
#   Total time: 362.87s
#   Total tokens: 129711
#   Throughput: 357.46 tokens/s
#   Avg latency per request: 3.629s
#   Batch size: 1
# ================================================================================

# 2025-10-17:09:42:39 INFO     [loggers.evaluation_tracker:209] Saving results aggregated
# qwen25_flash (model_name=Qwen/Qwen2.5-7B-Instruct,attn_implementation=eager,dtype_str=bfloat16,max_new_tokens=512,temperature=1.0,top_p=1.0,show_progress=True,show_speed=True), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.48|±  |0.0502|
# |     |       |strict-match    |     5|exact_match|↑  | 0.31|±  |0.0465|

# (.venv) ubuntu@prompt-raven:~/Fast-dLLM$ 