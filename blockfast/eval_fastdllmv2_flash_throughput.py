import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers_modelling.qwen_2_5_modelling import Fast_dLLM_QwenForCausalLM

torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'



def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y"}


@register_model("fast_dllm_v2_flash")
class FastDLLMv2FlashEvalHarness(LM):
    def __init__(
        self,
        model_path: str = "Efficient-Large-Model/Fast_dLLM_v2_7B",
        device: str = "cuda",
        dtype_str: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        max_new_tokens: int = 512,
        batch_size: int = 1,
        mask_id: int = 151665,
        use_block_cache: bool = False,
        small_block_size: int = 8,
        bd_size: int = 32,
        threshold: float = 0.9,
        temperature: float = 0.0,
        top_p: float = 0.95,
        show_progress: bool = True,
        show_speed: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_path = model_path
        self.attn_implementation = attn_implementation
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.mask_id = int(mask_id)
        self.use_block_cache = _to_bool(use_block_cache)
        self.small_block_size = int(small_block_size)
        self.bd_size = int(bd_size)
        self.threshold = float(threshold)
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
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

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
                prompts = []
                for req in batch_requests:
                    question = req.args[0]
                    if req.task_name.startswith("minerva_math"):
                        question = question.replace(
                            "Solution:", "Please reason step by step, and put your final answer within \\boxed{}."
                        )
                    elif req.task_name.startswith("gsm8k"):
                        question = question.replace(
                            "Answer:", "Please reason step by step, and put your final answer within \\boxed{}."
                        )
                    prompts.append(question)

                # Tokenize with padding
                batched_input_ids = []
                max_len = 0
                min_len = float("inf")
                seq_len = []

                for prompt in prompts:
                    model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
                    input_ids = model_inputs["input_ids"]
                    batched_input_ids.append(input_ids)
                    max_len = max(max_len, input_ids.shape[1])
                    min_len = min(min_len, input_ids.shape[1])
                    seq_len.append(input_ids.shape[1])

                # Pad batched_input_ids to the same length
                batched_input_ids = [
                    torch.cat(
                        [
                            input_ids,
                            torch.full(
                                (1, max_len - input_ids.shape[1]),
                                self.mask_id,
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ],
                        dim=1,
                    )
                    for input_ids in batched_input_ids
                ]
                batched_input_ids = torch.cat(batched_input_ids, dim=0)

                # Measure prefill time (first forward pass)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                prefill_start = time.time()

                # Do a single forward pass to measure prefill
                with torch.no_grad():
                    _ = self.model(batched_input_ids, use_cache=True)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                prefill_end = time.time()
                prefill_time = prefill_end - prefill_start

                # Measure decode time (full generation)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                decode_start = time.time()

                generated_ids = self.model.generate(
                    input_ids=batched_input_ids,
                    max_new_tokens=self.max_new_tokens,
                    mask_id=self.mask_id,
                    small_block_size=self.small_block_size,
                    block_size=self.bd_size,
                    use_block_cache=self.use_block_cache,
                    threshold=self.threshold,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                decode_end = time.time()
                decode_time = decode_end - decode_start

                # Update statistics
                batch_prefill_tokens = sum(seq_len)
                self.total_prefill_tokens += batch_prefill_tokens
                self.total_prefill_time += prefill_time

                # Decode each sample in batch
                for batch_idx, (input_len, gen_ids) in enumerate(zip(seq_len, generated_ids)):
                    continuation = gen_ids[input_len:]
                    continuation = continuation[continuation != self.mask_id]
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
                print(f"Throughput Metrics (Fast-dLLM v2 - {self.attn_implementation or 'default'})")
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
                print(
                    f"  Throughput: {(self.total_prefill_tokens + self.total_decode_tokens) / total_time:.2f} tokens/s"
                )
                print(f"  Avg latency per request: {total_time / len(requests):.3f}s")
                print(f"  Batch size: {self.batch_size}")
                print(f"{'=' * 80}\n")

        return output


def _build_cli_args(namespace, use_flash: bool) -> List[str]:
    attn_impl = "flash_attention_2" if use_flash else namespace.non_flash_attn_impl
    attn_label = "flash" if use_flash else namespace.non_flash_attn_impl

    model_args = [
        f"model_path={namespace.model_path}",
        f"attn_implementation={attn_impl}",
        f"dtype_str={namespace.dtype}",
        f"max_new_tokens={namespace.max_new_tokens}",
        f"mask_id={namespace.mask_id}",
        f"use_block_cache={namespace.use_block_cache}",
        f"small_block_size={namespace.small_block_size}",
        f"bd_size={namespace.bd_size}",
        f"threshold={namespace.threshold}",
        f"temperature={namespace.temperature}",
        f"top_p={namespace.top_p}",
        # Remove batch_size from here - it's passed separately by lm-eval
        "show_progress=True",
        "show_speed=True",
    ]

    output_path = Path(namespace.output_dir) / f"gsm8k_fast_dllm_v2_{attn_label}_bs{namespace.batch_size}.json"

    args = [
        "lm_eval",
        "--model",
        "fast_dllm_v2_flash",
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
    parser = argparse.ArgumentParser(
        description="Evaluate Fast-dLLM v2 on GSM8K with and without flash attention."
    )
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--mask_id", type=int, default=151665)
    parser.add_argument("--use_block_cache", action="store_true")
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--bd_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--non_flash_attn_impl",
        type=str,
        default="sdpa",
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
        print(f"Evaluating Fast-dLLM v2 on GSM8K with {label} attention")
        print("=" * 80 + "\n")

        sys.argv = _build_cli_args(args, use_flash)
        cli_evaluate()


if __name__ == "__main__":
    main()