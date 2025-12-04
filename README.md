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

## Quick Start

### Single Task Evaluation

```bash
# Generate and evaluate a single TileBench task
python scripts/generate_and_eval_single_sample_tilebench.py \
    level=basic \
    task_name=online_softmax \
    server_type=openai \
    model_name=gpt-4 \
    prompt_style=full
```

**Arguments**:
- `level`: Task difficulty (`basic`, `medium`, `advanced`)
- `task_name`: Name of the task (e.g., `online_softmax`, `layernorm`)
- `task_index`: Alternatively, use 0-based index instead of name
- `prompt_style`: Prompt complexity (`minimal`, `standard`, `full`)
- `server_type`: LLM provider (see `.env.example`)
- `model_name`: Model to use

### Batch Generation

Generate kernels for all tasks in a level:

```bash
python scripts/generate_samples_tilebench.py \
    run_name=gpt-5.1-basic-full \
    level=basic \
    server_type=openai \
    model_name=gpt-4 \
    prompt_style=full \
    num_workers=10
```

This creates a run directory: `runs/tilebench/{run_name}/`

### Batch Evaluation

Evaluate all generated kernels from a run:

```bash
python scripts/eval_from_generations_tilebench.py \
    run_name=gpt-5.1-basic-full \
    level=basic \
    num_workers=4
```

Optionally evaluate on full shape suites:
```bash
python scripts/eval_from_generations_tilebench.py \
    run_name=gpt-5.1-basic-full \
    level=basic \
    eval_on_suite=True \
    num_workers=4
```

## Key Differences from Original KernelBench

| Aspect | KernelBench | TileBench Adaptation |
|--------|-------------|---------------------|
| **Levels** | `level1`, `level2`, ... | `basic`, `medium`, `advanced` |
| **Task Format** | Single `.py` file | Directory with `reference.py` |
| **Prompts** | TOML-based templates | Per-task markdown files |
| **Target** | PyTorch → CUDA/Triton | Reference → TileLang |
| **Evaluation** | Against PyTorch model | Against `reference()` function |

## File Structure

```
baselines/KernelBench/
├── src/
│   ├── dataset_tilebench.py         # TileBench dataset loading
│   ├── prompt_constructor_tilebench.py  # Prompt construction from task files
│   ├── eval_tilebench.py            # Evaluation against reference.py
│   └── ...                          # Original KernelBench files
├── scripts/
│   ├── generate_and_eval_single_sample_tilebench.py
│   ├── generate_samples_tilebench.py
│   ├── eval_from_generations_tilebench.py
│   └── ...                          # Original KernelBench scripts
└── runs/tilebench/                  # Output directory
    └── {run_name}/
        ├── config.json
        ├── generation_results.json
        ├── eval_summary.json
        └── {task_name}/
            ├── prompt.txt
            ├── response.txt
            ├── generated_kernel.py
            └── eval_result.json
```

## Prompt Styles

TileBench supports three prompt styles:

1. **minimal**: Only core `prompt_codegen.md`
2. **standard**: Codegen + few-shot + correctness
3. **full**: All available prompt components

Each task may have different available components. The system automatically includes what's available.

## Evaluation Metrics

Similar to KernelBench, we compute:
- **Compilation rate**: Percentage of kernels that compile
- **Correctness rate**: Percentage that pass correctness checks
- **Speedup**: Ratio of reference time to kernel time
- **fast_p metrics**: Correctness with minimum speedup thresholds
  - `fast_0`: Correct (≥0x speedup)
  - `fast_1`: Correct and faster than reference (≥1x)
  - `fast_2`: Correct and ≥2x faster

## Advanced Usage

### Custom Task Selection

Generate only specific tasks:
```bash
python scripts/generate_samples_tilebench.py \
    run_name=selected_tasks \
    level=basic \
    tasks=online_softmax,layernorm,topk \
    ...
```

Skip specific tasks:
```bash
python scripts/generate_samples_tilebench.py \
    run_name=exclude_some \
    level=basic \
    skip_tasks=topk \
    ...
```

### Using Task Index

Use numeric index instead of name:
```bash
python scripts/generate_and_eval_single_sample_tilebench.py \
    level=basic \
    task_index=0 \
    ...
```

### Reasoning Models

For models like o1, o3, Gemini with extended thinking:
```bash
python scripts/generate_and_eval_single_sample_tilebench.py \
    level=basic \
    task_name=online_softmax \
    server_type=openai \
    model_name=o1-preview \
    is_reasoning_model=True \
    reasoning_effort=high
```

## Programmatic Usage

```python
from src.dataset_tilebench import get_task_by_name, construct_tilebench_dataset
from src.prompt_constructor_tilebench import construct_tilebench_prompt_simple
from src.eval_tilebench import eval_tilebench_kernel

# Get task info
task_info = get_task_by_name("online_softmax", level="basic")

# Construct prompt
prompt = construct_tilebench_prompt_simple(task_info, style="full")

# Generate kernel (your LLM code here)
kernel_source = your_llm(prompt)

# Evaluate
result = eval_tilebench_kernel(task_info, kernel_source)

print(f"Compiled: {result.compiled}")
print(f"Correct: {result.correctness}")
print(f"Speedup: {result.runtime_stats.get('speedup', 0):.2f}x")
```

## Dataset Statistics

To see all available tasks:
```bash
python -m src.dataset_tilebench
```

This will print:
- Number of tasks per level
- Task names
- Available prompt files
- Whether reference kernels exist

## Comparison to Original TileBench Workflow

**Original TileBench**: Tasks are developed and tested individually with manual prompting.

**This Baseline**: Systematic evaluation of LLMs across all TileBench tasks with:
- Consistent prompting across tasks
- Automated generation and evaluation
- Batch processing for multiple models/configurations
- Performance metrics and analysis

## Troubleshooting

### Import Errors
Make sure you're in the repo root and have installed: `pip install -e .`

### Task Not Found
Check task name matches directory name exactly. Use `python -m src.dataset_tilebench` to list all tasks.

### Evaluation Failures
- Check that generated kernel has a callable entry point
- Ensure TileLang is properly installed
- Check error messages in `eval_result.json`

### CUDA Out of Memory
Reduce batch size in evaluation or run tasks sequentially (`num_workers=1`)

## Citation

If you use this baseline, please cite both KernelBench and TileBench:

```bibtex
@misc{ouyang2025kernelbench,
  title={KernelBench: Can LLMs Write Efficient GPU Kernels?}, 
  author={Anne Ouyang and Simon Guo and Simran Arora and Alex L. Zhang and William Hu and Christopher Ré and Azalia Mirhoseini},
  year={2025},
  eprint={2502.10517},
  archivePrefix={arXiv},
}

# TileBench citation (add when available)
```

## Contributing

This is an adaptation of KernelBench for TileBench. For issues:
- TileBench-specific: Open issue in TileBench repo
- KernelBench infrastructure: Open issue in KernelBench repo

## License

Same as KernelBench (MIT License)

