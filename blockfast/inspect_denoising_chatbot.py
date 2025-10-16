from time import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random
from my_transformers_modelling.g_qwen_2_5_modelling import Fast_dLLM_QwenForCausalLM

model_name = "Efficient-Large-Model/Fast_dLLM_v2_7B"

model = Fast_dLLM_QwenForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# List of questions to test
questions = [
    "What are the key differences between Python and JavaScript?",
    "Explain the concept of machine learning in simple terms",
    "How does photosynthesis work?",
    "What is the capital of France and what are its main attractions?"
]

# Store results for both methods
results = {
    "old": {"throughputs": [], "times": [], "responses": []},
    "new": {"throughputs": [], "times": [], "responses": []}
}

print("=" * 80)
print("RUNNING BENCHMARKS")
print("=" * 80)

for method_name, use_old_method in [("old", True), ("new", False)]:
    print(f"\n{'='*80}")
    print(f"Testing with {method_name.upper()} method")
    print(f"{'='*80}\n")
    
    for i, prompt in enumerate(questions, 1):
        fix_seed(42)  # Reset seed for each question for fair comparison
        
        messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print(f"Question {i}: {prompt}")
        
        start_time = time()
        generated_ids = model.generate(
            model_inputs["input_ids"],
            tokenizer=tokenizer,
            max_new_tokens=512,
            small_block_size=8,
            threshold=0.9,
            use_old_method=use_old_method,
        )
        end_time = time()
        
        num_generated_tokens = generated_ids.shape[1] - model_inputs["input_ids"].shape[1]
        generation_time = end_time - start_time
        tokens_per_second = num_generated_tokens / generation_time
        
        response = tokenizer.decode(generated_ids[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        results[method_name]["throughputs"].append(tokens_per_second)
        results[method_name]["times"].append(generation_time)
        results[method_name]["responses"].append(response)
        
        print(f"  Time: {generation_time:.2f}s | Tokens: {num_generated_tokens} | Throughput: {tokens_per_second:.2f} tok/s")
        print("-" * 80)

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

for method_name in ["old", "new"]:
    avg_throughput = np.mean(results[method_name]["throughputs"])
    avg_time = np.mean(results[method_name]["times"])
    std_throughput = np.std(results[method_name]["throughputs"])
    
    print(f"\n{method_name.upper()} Method:")
    print(f"  Average throughput: {avg_throughput:.2f} ± {std_throughput:.2f} tokens/s")
    print(f"  Average time: {avg_time:.2f}s")
    print(f"  Min throughput: {min(results[method_name]['throughputs']):.2f} tokens/s")
    print(f"  Max throughput: {max(results[method_name]['throughputs']):.2f} tokens/s")

speedup = np.mean(results["new"]["throughputs"]) / np.mean(results["old"]["throughputs"])
print(f"\nSpeedup (new vs old): {speedup:.2f}x")

# Print response comparisons
print("\n" + "=" * 80)
print("RESPONSE COMPARISONS")
print("=" * 80)

for i, question in enumerate(questions):
    print(f"\n{'='*80}")
    print(f"Question {i+1}: {question}")
    print(f"{'='*80}")
    
    print(f"\nOLD Method Response:")
    print(f"{'-'*80}")
    print(results["old"]["responses"][i])
    
    print(f"\n\nNEW Method Response:")
    print(f"{'-'*80}")
    print(results["new"]["responses"][i])
    print("\n")