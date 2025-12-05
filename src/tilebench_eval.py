"""
TileBench Evaluation Wrapper for KernelBench Infrastructure

This module provides a bridge between KernelBench's evaluation infrastructure
and TileBench's reference.py based evaluation system.
"""

import os
from typing import Optional
import torch

from src.tilebench_reference import (
    load_tilebench_reference,
    get_reference_info,
    format_reference_as_problem_description,
)
from src.eval_cuda_against_tilebench import (
    eval_kernel_against_tilebench,
    eval_kernel_on_all_shapes,
    get_aggregate_result,
    KernelEvalResult,
)
from src.eval import KernelExecResult


def convert_tilebench_to_kernelbench_result(tilebench_result: KernelEvalResult) -> KernelExecResult:
    """
    Convert TileBench evaluation result to KernelBench format.
    
    Args:
        tilebench_result: Result from TileBench evaluation
        
    Returns:
        KernelExecResult compatible with KernelBench infrastructure
    """
    runtime_stats = {}
    if tilebench_result.runtime_ms > 0:
        runtime_stats = {
            "kernel_time_ms": tilebench_result.runtime_ms,
            "reference_time_ms": tilebench_result.reference_runtime_ms,
            "speedup": tilebench_result.speedup,
        }
    
    metadata = dict(tilebench_result.metadata)
    if tilebench_result.error_type:
        metadata["error_type"] = tilebench_result.error_type
    if tilebench_result.error_message:
        metadata["error_message"] = tilebench_result.error_message
    
    return KernelExecResult(
        compiled=tilebench_result.compiled,
        correctness=tilebench_result.correctness,
        runtime=tilebench_result.runtime_ms,
        runtime_stats=runtime_stats,
        metadata=metadata,
    )


def eval_kernel_against_tilebench_ref(
    reference_path: str,
    custom_kernel_src: str,
    backend: str = "cuda",
    verbose: bool = False,
    measure_performance: bool = True,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    precision: Optional[torch.dtype] = None,
    **kwargs,
) -> KernelExecResult:
    """
    Evaluate a generated kernel against TileBench reference.
    
    This function provides a KernelBench-compatible interface to TileBench evaluation.
    
    Args:
        reference_path: Path to TileBench reference.py file
        custom_kernel_src: Generated kernel source code
        backend: "cuda", "triton", or "python"
        verbose: Enable verbose output
        measure_performance: Whether to measure performance
        num_correct_trials: Number of correctness trials (not used for TileBench)
        num_perf_trials: Number of performance trials
        precision: Torch dtype for computation (not used for TileBench)
        **kwargs: Additional arguments
        
    Returns:
        KernelExecResult with evaluation metrics
    """
    # Determine warmup and repeat from num_perf_trials
    warmup = 3
    repeat = max(num_perf_trials // 5, 10)  # Use 1/5 of perf trials, minimum 10
    
    try:
        # Run TileBench evaluation
        tilebench_result = eval_kernel_against_tilebench(
            reference_path=reference_path,
            kernel_source=custom_kernel_src,
            backend=backend,
            kernel_name="run",
            config=None,  # Use default config
            warmup=warmup,
            repeat=repeat,
            test_all_shapes=False,
        )
        
        # Convert to KernelBench format
        result = convert_tilebench_to_kernelbench_result(tilebench_result)
        
        if verbose:
            print(f"TileBench Evaluation:")
            print(f"  Compiled: {result.compiled}")
            print(f"  Correct: {result.correctness}")
            if measure_performance and result.runtime > 0:
                print(f"  Runtime: {result.runtime:.4f} ms")
                if "speedup" in result.runtime_stats:
                    print(f"  Speedup: {result.runtime_stats['speedup']:.2f}x")
        
        return result
        
    except Exception as e:
        # Handle any errors and return a failed result
        return KernelExecResult(
            compiled=False,
            correctness=False,
            runtime=-1.0,
            runtime_stats={},
            metadata={
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )


def get_tilebench_problem_info(reference_path: str) -> dict:
    """
    Extract problem information from TileBench reference.
    
    Args:
        reference_path: Path to reference.py
        
    Returns:
        Dict with problem information
    """
    return get_reference_info(reference_path)


def get_tilebench_problem_description(reference_path: str) -> str:
    """
    Get formatted problem description for prompting.
    
    Args:
        reference_path: Path to reference.py
        
    Returns:
        Formatted problem description string
    """
    return format_reference_as_problem_description(reference_path)


def is_tilebench_reference(file_path: str) -> bool:
    """
    Check if a file path is a TileBench reference.py file.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file is a TileBench reference.py
    """
    return file_path.endswith("reference.py") and "TileBench-Benchmark" in file_path

