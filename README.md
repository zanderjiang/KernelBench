# KernelBench for TileBench

This is an adapted version of KernelBench infrastructure to evaluate LLMs on TileBench tasks. TileBench focuses on TileLang kernel generation for GPU operators.

## Overview

### TileBench Structure
- **Levels**: `basic/`, `medium/`, `advanced/` (instead of level1, level2, etc.)
- **Tasks**: Each task is a directory containing:
  - `reference.py`: PyTorch reference implementation with standard API
  - `prompt_codegen.md`: Core generation prompt
  - `prompt_fewshot.md`: Few-shot examples (optional)
  - `prompt_correctness.md`: Correctness guidelines (optional)
  - `prompt_performance.md`: Performance tips (optional)
  - Other prompt files for comprehensive guidance

### TileBench Reference API
Each `reference.py` provides:
```python
def description() -> str
def get_default_config() -> dict
def make_inputs(cfg) -> dict
def reference(**inputs) -> torch.Tensor
def check(y_ref, y_out, atol, rtol) -> None
def get_shape_suites() -> list[dict]
```

## Installation

```bash
conda create --name tile-bench-baseline python=3.10
conda activate kernel-bench
pip install -r requirements.txt
pip install -e .
```
# TileBench + KernelBench Quick Start Guide

## Setup

```bash
# Activate the conda environment
conda activate tile-bench-baseline

# Verify integration works
cd /home/akj2/TileBench/baselines/KernelBench
python3 test_tilebench_integration.py
```

## Basic Usage

### 1. Generate and Evaluate Single Problem

Test with LayerNorm (problem_id=1 in basic level):

```bash
cd /home/akj2/TileBench/baselines/KernelBench

conda run -n tile-bench-baseline python3 scripts/generate_and_eval_single_sample.py \
    dataset_src=local \
    level=basic \
    problem_id=1 \
    eval_mode=local \
    server_type=openai \
    model_name=gpt-4o \
    max_tokens=8192 \
    temperature=0.0 \
    backend=cuda \
    precision=fp32 \
    verbose=True
```

**Output Location:**
- Prompt: `results/eval_logs/prompt_level_basic_problem_1.txt`
- Generated Kernel: `results/eval_logs/generated_kernel_level_basic_problem_1.py`
- Evaluation Result: `results/eval_logs/eval_result_level_basic_problem_1.txt`

### 2. Batch Generate Kernels

Generate kernels for all basic level problems:

```bash
conda run -n tile-bench-baseline python3 scripts/generate_samples.py \
    dataset_src=local \
    level=basic \
    run_name=gpt4_basic_run1 \
    server_type=openai \
    model_name=gpt-4o \
    backend=cuda \
    num_samples=1 \
    num_workers=4 \
    log_prompt=True
```

**Output Structure:**
```
runs/gpt4_basic_run1/
├── generation_config.yaml
├── level_basic_problem_1_sample_0_kernel.py
├── level_basic_problem_1_sample_0_prompt.txt
├── level_basic_problem_2_sample_0_kernel.py
└── ...
```

### 3. Batch Evaluate Generated Kernels

Evaluate all generated kernels:

```bash
conda run -n tile-bench-baseline python3 scripts/eval_from_generations.py \
    dataset_src=local \
    level=basic \
    run_name=gpt4_basic_run1 \
    eval_mode=local \
    gpu_arch='["Ada"]' \
    num_gpu_devices=1 \
    backend=cuda \
    precision=fp32 \
    num_correct_trials=5 \
    num_perf_trials=100 \
    verbose=True
```

**Output:**
- `runs/gpt4_basic_run1/eval_results.json`

### 4. View Results

```bash
# Pretty print evaluation results
python3 -c "
import json
with open('runs/gpt4_basic_run1/eval_results.json') as f:
    results = json.load(f)
    for problem_id, evals in results.items():
        for e in evals:
            print(f\"Problem {problem_id}, Sample {e['sample_id']}\")
            print(f\"  Compiled: {e['compiled']}\")
            print(f\"  Correct: {e['correctness']}\")
            if 'runtime' in e and e['runtime'] > 0:
                print(f\"  Runtime: {e['runtime']:.4f} ms\")
            print()
"
```

## Problem Discovery

### List All Available Problems

```python
from src.dataset import construct_tilebench_dataset

# List basic level problems
basic_problems = construct_tilebench_dataset("basic")
for i, path in enumerate(basic_problems, 1):
    import os
    problem_name = os.path.basename(os.path.dirname(path))
    print(f"Problem {i}: {problem_name}")
```

### Current TileBench Problems

**Basic Level (4 problems):**
1. layernorm
2. online_softmax
3. rmsnorm
4. topk

**Medium Level (6 problems):**
1. conv2d
2. flash_attention_bwd
3. flash_attention_fwd
4. gemm_with_epilogue
5. grouped_gemm
6. hadamard

**Advanced Level (11 problems):**
1. attention_sink
2. blocksparse_gemm
3. cross_entropy
4. decode_attention
5. mla_decode
6. moe_gemm
7. prefix_sum
8. rotary_embedding
9. selective_scan
10. sparse_attention
11. sparse_mla_ws

## Level Comparison

| Aspect | Basic | Medium | Advanced |
|--------|-------|--------|----------|
| Problems | 4 | 6 | 11 |
| Complexity | Single operations | Fused operations | Complex patterns |
| Examples | LayerNorm, RMSNorm | Flash Attention, Conv2D | MoE, Sparse Attention |

## Using Different LLM Providers

### OpenAI (GPT-4, GPT-4o)

```bash
server_type=openai \
model_name=gpt-4o \
max_tokens=8192 \
temperature=0.0
```

### Google (Gemini)

```bash
server_type=google \
model_name=gemini/gemini-2.5-flash \
max_tokens=8192 \
temperature=0.0
```

### Anthropic (Claude)

```bash
server_type=anthropic \
model_name=claude-3-5-sonnet-20241022 \
max_tokens=8192 \
temperature=0.0
```

### Local Models (via OpenAI-compatible API)

```bash
server_type=custom \
model_name=deepseek-ai/DeepSeek-V3 \
api_base=http://localhost:8000/v1 \
max_tokens=8192 \
temperature=0.0
```

## Backend Options

### CUDA (Default)

```bash
backend=cuda
```

Generated kernel must use `torch.utils.cpp_extension.load_inline()` with CUDA C++ code.

### Triton

```bash
backend=triton
```

Generated kernel must use `@triton.jit` decorator.

### Python (Reference/Testing)

```bash
backend=python
```

Generated kernel is pure PyTorch code (useful for testing prompt/eval pipeline).

## Advanced Options

### Multiple Samples per Problem (for pass@k)

```bash
# Generate 10 samples per problem
python3 scripts/generate_samples.py \
    ... \
    num_samples=10

# Evaluate all samples
python3 scripts/eval_from_generations.py \
    ... \
    num_samples_per_problem=10 \
    pass_at_k_values='[1,5,10]'
```

### Cloud Evaluation with Modal

```bash
# Requires Modal account and setup
python3 scripts/eval_from_generations.py \
    dataset_src=local \
    level=basic \
    run_name=gpt4_basic_run1 \
    eval_mode=modal \
    gpu=A10G \
    backend=cuda
```

### Include Hardware Information in Prompt

```bash
python3 scripts/generate_and_eval_single_sample.py \
    ... \
    include_hardware_info=True \
    hardware_gpu_name=A100
```

### Custom GPU Architecture

```bash
# For specific GPU architectures
gpu_arch='["Ampere"]'  # A100, A10, A40
gpu_arch='["Ada"]'     # RTX 4090, L40S, L4
gpu_arch='["Hopper"]'  # H100
```

## Troubleshooting

### Issue: "No module named 'litellm'"

**Solution:** Make sure you're using the conda environment:
```bash
conda activate tile-bench-baseline
```

### Issue: "CUDA device not available"

**Solution:** Check CUDA setup:
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
nvcc --version
```

### Issue: "Compilation failed"

**Common causes:**
1. Generated code has syntax errors
2. CUDA toolkit version mismatch
3. Missing headers or libraries

**Debug:**
- Check generated kernel: `results/eval_logs/generated_kernel_level_*_problem_*.py`
- Add `verbose=True` to see compilation errors
- Try `backend=python` first to test logic

### Issue: "Outputs differ"

**Common causes:**
1. Numerical precision (try `precision=fp32`)
2. Incorrect implementation
3. Wrong dimension handling

**Debug:**
- Check reference implementation in `TileBench-Benchmark/{level}/{problem}/reference.py`
- Compare shapes: Add debug prints in generated kernel
- Test with smaller inputs

## Example: Complete Workflow

```bash
# 1. Test single problem first
conda run -n tile-bench-baseline python3 scripts/generate_and_eval_single_sample.py \
    dataset_src=local \
    level=basic \
    problem_id=1 \
    server_type=openai \
    model_name=gpt-4o \
    backend=cuda \
    verbose=True

# 2. If successful, batch generate
conda run -n tile-bench-baseline python3 scripts/generate_samples.py \
    dataset_src=local \
    level=basic \
    run_name=my_experiment \
    server_type=openai \
    model_name=gpt-4o \
    backend=cuda \
    num_samples=1

# 3. Batch evaluate
conda run -n tile-bench-baseline python3 scripts/eval_from_generations.py \
    dataset_src=local \
    level=basic \
    run_name=my_experiment \
    eval_mode=local \
    backend=cuda

# 4. Check results
cat runs/my_experiment/eval_results.json
```

- Test suite: `test_tilebench_integration.py`
- TileBench problems: `/home/akj2/TileBench/TileBench-Benchmark/`
- Generated runs: `runs/`
- Logs: `results/eval_logs/`