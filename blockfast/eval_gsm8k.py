import argparse
import sys
import time
from typing import List
from pathlib import Path

import accelerate
import numpy as np
import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer
from tqdm import tqdm

from my_transformers_modelling.g_qwen_2_5_modelling import Fast_dLLM_QwenForCausalLM


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y"}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@register_model("fast_dllm_blockfast")
class Fast_dLLMBlockfastEvalHarness(LM):
    def __init__(
        self,
        model_path: str = "Efficient-Large-Model/Fast_dLLM_v2_7B",
        device: str = "cuda",
        show_speed: bool = False,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        mask_id: int = 151665,
        stop_token: int = 151645,
        block_size: int = 32,
        small_block_size: int = 8,
        threshold: float = 0.9,
        temperature: float = 0.0,
        top_p: float = 0.95,
        use_block_cache: bool = False,
        use_old_method: bool = False,
        show_outputs: bool = False,
        show_progress: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.max_new_tokens = int(max_new_tokens)
        self.mask_id = int(mask_id)
        self.stop_token = int(stop_token)
        self.block_size = int(block_size)
        self.small_block_size = int(small_block_size)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.use_block_cache = _to_bool(use_block_cache)
        self.show_speed = _to_bool(show_speed)
        self.use_old_method = _to_bool(use_old_method)
        self.batch_size = int(batch_size)
        self.model_path = model_path
        self._world_size = 1
        self._rank = 0
        self.show_outputs = _to_bool(show_outputs)
        self.show_progress = _to_bool(show_progress)

        if seed is not None:
            set_seed(int(seed))

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if self.accelerator is not None:
            model_kwargs["device_map"] = {"": f"{self.accelerator.device}"}
        else:
            model_kwargs["device_map"] = "auto"

        self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
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
        return self.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests) -> List[str]:
        output = [None] * len(requests)
        total_generated_tokens = 0
        total_generation_time = 0.0

        progress_bar = None
        if self.rank == 0 and self.show_progress:
            progress_bar = tqdm(total=len(requests), desc="Generating", leave=False)

        for idx, req in enumerate(requests):
            prompt = req.args[0]
            if req.task_name.startswith("gsm8k"):
                prompt = prompt.replace(
                    "Answer:",
                    "Please reason step by step, and put your final answer within \\boxed{}.",
                )

            model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = model_inputs["input_ids"]
            input_len = input_ids.shape[1]

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.time()
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                mask_id=self.mask_id,
                threshold=self.threshold,
                small_block_size=self.small_block_size,
                block_size=self.block_size,
                stop_token=self.stop_token,
                temperature=self.temperature,
                top_p=self.top_p,
                use_block_cache=self.use_block_cache,
                use_old_method=self.use_old_method,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            end = time.time()

            continuation = generated_ids[0, input_len:]
            continuation = continuation[(continuation != self.mask_id) & (continuation != self.stop_token)]
            total_generated_tokens += continuation.numel()
            total_generation_time += end - start

            decoded = self.tokenizer.decode(continuation, skip_special_tokens=True)
            output[idx] = decoded.strip()

            if progress_bar is not None:
                progress_bar.update(1)

            if self.rank == 0 and self.show_outputs:
                print("=" * 20)
                print("prompt:", prompt)
                print("answer:", output[idx])
                print("=" * 20, end="\n\n")

        if progress_bar is not None:
            progress_bar.close()

        if self.show_speed and self.rank == 0 and total_generation_time > 0:
            throughput = total_generated_tokens / total_generation_time
            print(f"Total generated tokens: {total_generated_tokens}")
            print(f"Total generation time: {total_generation_time:.2f}s")
            print(f"Throughput: {throughput:.2f} tokens/s")

        return output


def _build_cli_args(namespace, use_old_method: bool) -> List[str]:
    model_args = [
        f"model_path={namespace.model_path}",
        f"use_old_method={str(use_old_method)}",
        f"max_new_tokens={namespace.max_new_tokens}",
        f"mask_id={namespace.mask_id}",
        f"block_size={namespace.block_size}",
        f"small_block_size={namespace.small_block_size}",
        f"threshold={namespace.threshold}",
        f"temperature={namespace.temperature}",
        f"top_p={namespace.top_p}",
        f"use_block_cache={str(namespace.use_block_cache)}",
        f"show_outputs={str(namespace.show_outputs)}",
        f"show_progress={str(not namespace.no_progress_bar)}",
        "show_speed=True",
    ]
    method_label = "old" if use_old_method else "new"
    output_path = Path(namespace.output_dir) / f"gsm8k_{method_label}.json"

    args = [
        "lm_eval",
        "--model",
        "fast_dllm_blockfast",
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
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM8K eval for old and new decoding methods.")
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--mask_id", type=int, default=151665)
    parser.add_argument("--use_block_cache", action="store_true")
    parser.add_argument("--methods", choices=["old", "new", "both"], default="both")
    parser.add_argument("--show_outputs", action="store_true")
    parser.add_argument("--no_progress_bar", action="store_true")
    parser.add_argument("--output_dir", type=str, default="logs")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    methods = ["old", "new"] if args.methods == "both" else [args.methods]

    for method in methods:
        use_old_method = method == "old"
        label = "OLD" if use_old_method else "NEW"
        if method != methods[0] and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if label:
            print("\n" + "=" * 80)
            print(f"Running GSM8K with {label} method")
            print("=" * 80 + "\n")

        sys.argv = _build_cli_args(args, use_old_method)
        cli_evaluate()


if __name__ == "__main__":
    main()
