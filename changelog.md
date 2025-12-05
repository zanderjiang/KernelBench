# TileBench Integration for KernelBench

This document describes the integration of TileBench's reference-based evaluation system with KernelBench script baseline.

## Overview

The integration allows KernelBench to use TileBench's `reference.py` files as problem datasets while maintaining full compatibility with KernelBench's existing workflow for:
- Prompt construction
- Kernel generation via LLMs
- CUDA kernel compilation and evaluation
- Performance benchmarking

## Key Changes

### 1. Dataset Structure

**TileBench Structure:**
```
TileBench-Benchmark/
├── basic/
│   ├── layernorm/
│   │   └── reference.py
│   ├── rmsnorm/
│   │   └── reference.py
│   └── ...
├── medium/
│   └── ...
└── advanced/
    └── ...
```

**Level Mapping:**
- `basic` → Level 1 concepts
- `medium` → Level 2 concepts  
- `advanced` → Level 3 concepts

### 2. New Modules

#### `src/tilebench_reference.py`
- `load_tilebench_reference()`: Load reference.py module
- `get_reference_info()`: Extract problem metadata
- `format_reference_as_problem_description()`: Format for prompts

#### `src/eval_cuda_against_tilebench.py`
- `eval_kernel_against_tilebench()`: Evaluate kernel against TileBench reference
- `load_generated_kernel()`: Load CUDA/Triton/Python kernels
- `KernelEvalResult`: Evaluation result format

#### `src/tilebench_eval.py`
- `eval_kernel_against_tilebench_ref()`: KernelBench-compatible wrapper
- `convert_tilebench_to_kernelbench_result()`: Format conversion
- `is_tilebench_reference()`: Detect TileBench files

#### `src/dataset.py` (extended)
- `construct_tilebench_dataset()`: Discover TileBench problems
- `get_tilebench_level_mapping()`: Map numeric levels to names
- `construct_dataset_unified()`: Unified dataset loader

### 3. Modified Scripts

All three main scripts now support TileBench:
- `scripts/generate_and_eval_single_sample.py`
- `scripts/generate_samples.py`
- `scripts/eval_from_generations.py`

## Usage

### Single Sample Generation and Evaluation

Generate and evaluate a single TileBench problem:

```bash
python3 scripts/generate_and_eval_single_sample.py \
    dataset_src=local \
    level=basic \
    problem_id=1 \
    eval_mode=local \
    server_type=google \
    model_name=gemini/gemini-2.5-flash \
    max_tokens=8192 \
    temperature=0.0 \
    backend=cuda \
    precision=fp32
```

**Key Parameters:**
- `dataset_src=local`: Always use local for TileBench
- `level`: String level name (`basic`, `medium`, `advanced`)
- `problem_id`: 1-indexed problem number within that level
- `backend`: `cuda`, `triton`, or `python`

### Batch Generation

Generate kernels for multiple problems:

```bash
python3 scripts/generate_samples.py \
    dataset_src=local \
    level=basic \
    run_name=tilebench_basic_gpt4 \
    server_type=openai \
    model_name=gpt-4 \
    backend=cuda \
    num_samples=1 \
    subset='(1,10)'
```

This generates kernels for problems 1-10 in the `basic` level.

**Output Structure:**
```
runs/tilebench_basic_gpt4/
├── generation_config.yaml
├── level_basic_problem_1_sample_0_kernel.py
├── level_basic_problem_2_sample_0_kernel.py
└── ...
```

### Batch Evaluation

Evaluate previously generated kernels:

```bash
python3 scripts/eval_from_generations.py \
    dataset_src=local \
    level=basic \
    run_name=tilebench_basic_gpt4 \
    eval_mode=local \
    gpu_arch='["Ada"]' \
    num_gpu_devices=1 \
    backend=cuda \
    precision=fp32 \
    num_correct_trials=5 \
    num_perf_trials=100
```

**Output:**
- `runs/tilebench_basic_gpt4/eval_results.json`: Evaluation metrics

### Modal Cloud Evaluation

For cloud-based evaluation:

```bash
python3 scripts/eval_from_generations.py \
    dataset_src=local \
    level=basic \
    run_name=tilebench_basic_gpt4 \
    eval_mode=modal \
    gpu=A10G \
    backend=cuda
```

## Expected Kernel Format

Generated kernels must implement a `run()` function that matches TileBench's reference signature:

### Example: RMSNorm

**Input (from reference.py):**
```python
def run(A: Tensor[M, N]) -> Tensor[M, N]:
    """
    Implement RMSNorm: y = x / sqrt(mean(x^2) + eps)
    """
    # Your CUDA/Triton implementation
    return output
```

**Generated Kernel Structure (CUDA):**

```python
# CUDA kernel implementation with load_inline

def run(A):
    import torch
    from torch.utils.cpp_extension import load_inline
    
    cuda_source = """
    __global__ void rmsnorm_kernel(...) {
        // CUDA implementation
    }
    """
    
    cpp_source = """
    torch::Tensor rmsnorm_forward(...) {
        // Launch kernel
    }
    """
    
    module = load_inline(
        name='rmsnorm',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['rmsnorm_forward'],
        with_cuda=True,
    )
    
    return module.rmsnorm_forward(A)
```

## Evaluation Flow

1. **Load Reference**: Load TileBench `reference.py` using `load_tilebench_reference()`
2. **Generate Inputs**: Call `reference.make_inputs(config)` to create test tensors
3. **Get Ground Truth**: Call `reference.reference(**inputs)` for expected output
4. **Load Generated Kernel**: Parse and compile the LLM-generated kernel via `load_inline`
5. **Execute Kernel**: Call `kernel.run(**inputs)` to get actual output
6. **Check Correctness**: Compare outputs with `torch.allclose()` or `reference.check()`
7. **Benchmark Performance**: Time both reference and kernel with CUDA events

## Configuration Notes

### Problem ID Mapping

TileBench problems are discovered by scanning directories:

```python
# List all problems in basic level
from src.dataset import construct_tilebench_dataset
problems = construct_tilebench_dataset("basic")
# problems[0] -> first problem (problem_id=1)
# problems[1] -> second problem (problem_id=2)
```

Problems are sorted alphabetically by folder name (excluding `example-*`).

### Tolerances

Default tolerances for correctness:
- `atol = 1e-2` (absolute tolerance)
- `rtol = 1e-2` (relative tolerance)

Can be overridden in `reference.py`'s `get_default_config()`.

### Performance Metrics

The evaluation returns:
- `compiled`: Whether kernel compiled successfully
- `correctness`: Whether output matches reference
- `runtime_ms`: Average kernel execution time
- `reference_runtime_ms`: Average reference execution time
- `speedup`: `reference_runtime_ms / runtime_ms`

## Backward Compatibility

The integration maintains full backward compatibility with KernelBench:

```bash
# Original KernelBench usage still works
python3 scripts/generate_and_eval_single_sample.py \
    dataset_src=local \
    level=1 \
    problem_id=1 \
    ...
```

Numeric levels (1-4) continue to use `KernelBench/level{N}` directories.

## Troubleshooting

### "Reference not found" Error

Ensure TileBench-Benchmark is in the correct location:
```
/home/akj2/TileBench/TileBench-Benchmark/
```

### "Kernel compilation failed"

Check that:
1. Generated code is valid CUDA/Triton
2. `load_inline` is used correctly
3. CUDA toolkit is installed (`nvcc --version`)
4. PyTorch CUDA is available (`torch.cuda.is_available()`)

### "Outputs differ"

Common causes:
- Numerical precision issues (use `fp32` or adjust tolerances)
- Boundary condition bugs
- Race conditions in parallel code
- Incorrect dimension handling

### Modal GPU Attachment Failures

Modal evaluation includes automatic retry logic for GPU attachment failures. If issues persist:
- Check Modal quota limits
- Try different GPU types (`gpu=L40S`, `gpu=A10G`, etc.)
- Verify Modal authentication

## Directory Structure

```
baselines/KernelBench/
├── src/
│   ├── dataset.py                      # (extended) Dataset loaders
│   ├── tilebench_reference.py          # (new) TileBench reference loader
│   ├── eval_cuda_against_tilebench.py  # (new) TileBench evaluation
│   ├── tilebench_eval.py               # (new) Integration wrapper
│   ├── eval.py                         # KernelBench evaluation
│   ├── prompt_constructor_toml.py      # Prompt builder (unchanged)
│   └── ...
├── scripts/
│   ├── generate_and_eval_single_sample.py  # (modified)
│   ├── generate_samples.py                 # (modified)
│   └── eval_from_generations.py            # (modified)
└── TILEBENCH_INTEGRATION.md            # (new) This file
```

## Examples

### Example 1: Test LayerNorm on basic level

```bash
# Generate kernel
python3 scripts/generate_and_eval_single_sample.py \
    dataset_src=local \
    level=basic \
    problem_id=1 \
    backend=cuda \
    server_type=openai \
    model_name=gpt-4o

# Output shows:
# - Generated kernel saved to results/eval_logs/
# - Correctness: True/False
# - Speedup: X.XXx
```

### Example 2: Batch evaluate 10 problems

```bash
# Step 1: Generate all kernels
python3 scripts/generate_samples.py \
    dataset_src=local \
    level=basic \
    run_name=test_run \
    backend=cuda \
    subset='(1,10)'

# Step 2: Evaluate all
python3 scripts/eval_from_generations.py \
    dataset_src=local \
    level=basic \
    run_name=test_run \
    eval_mode=local

# Step 3: Check results
cat runs/test_run/eval_results.json
```

## API Reference

### TileBench Evaluation Function

```python
from src.tilebench_eval import eval_kernel_against_tilebench_ref

result = eval_kernel_against_tilebench_ref(
    reference_path="/path/to/reference.py",
    custom_kernel_src="<generated kernel code>",
    backend="cuda",  # or "triton", "python"
    measure_performance=True,
    num_perf_trials=100,
    verbose=False,
)

print(f"Compiled: {result.compiled}")
print(f"Correct: {result.correctness}")
print(f"Speedup: {result.runtime_stats.get('speedup', 0):.2f}x")
```