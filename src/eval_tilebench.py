"""
Evaluation helpers for TileBench tasks

TileBench reference.py structure:
- description(): Returns task description
- get_default_config(): Returns default configuration dict
- make_inputs(cfg): Creates input tensors
- reference(inputs): Reference implementation
- check(y_ref, y_out, atol, rtol): Correctness check
- get_shape_suites(): Returns list of test configurations
"""

import importlib
import importlib.util
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel

from src.utils import read_file


class TileBenchEvalResult(BaseModel):
    """Evaluation result for a TileBench task"""
    
    compiled: bool = False
    correctness: bool = False
    runtime: float = -1.0  # in milliseconds
    runtime_stats: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def load_reference_module(reference_path: str):
    """
    Load the reference.py module dynamically.
    
    Args:
        reference_path: Path to reference.py
        
    Returns:
        Loaded module object
    """
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    
    # Create a unique module name based on the path
    module_name = f"tilebench_ref_{os.path.basename(os.path.dirname(reference_path))}"
    
    spec = importlib.util.spec_from_file_location(module_name, reference_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def load_kernel_module(kernel_source: str, entry_point: str = "kernel_forward"):
    """
    Load generated kernel code dynamically.
    
    Args:
        kernel_source: Source code of the generated kernel
        entry_point: Name of the main kernel function to call
        
    Returns:
        Tuple of (module, kernel_function)
    """
    # Create a temporary file with the kernel source
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(kernel_source)
        tempfile_path = tmp_file.name
    
    try:
        # Import the module
        spec = importlib.util.spec_from_file_location("tilebench_kernel", tempfile_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["tilebench_kernel"] = module
        spec.loader.exec_module(module)
        
        # Get the entry point function
        if not hasattr(module, entry_point):
            # Try to find a callable kernel function
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and not attr_name.startswith("_"):
                    entry_point = attr_name
                    break
        
        kernel_fn = getattr(module, entry_point, None)
        
        return module, kernel_fn
    
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tempfile_path)
        except Exception:
            pass


def eval_tilebench_kernel(
    task_info: Dict,
    kernel_source: str,
    config: Optional[Dict] = None,
    warmup: int = 3,
    repeat: int = 20,
    check_correctness: bool = True,
    measure_performance: bool = True,
) -> TileBenchEvalResult:
    """
    Evaluate a generated kernel against the TileBench reference.
    
    Args:
        task_info: Task information dictionary
        kernel_source: Generated kernel source code
        config: Optional test configuration (uses default if None)
        warmup: Number of warmup runs for benchmarking
        repeat: Number of benchmark repetitions
        check_correctness: Whether to check correctness
        measure_performance: Whether to measure performance
        
    Returns:
        TileBenchEvalResult with evaluation metrics
    """
    result = TileBenchEvalResult()
    
    try:
        # Load reference module
        ref_module = load_reference_module(task_info["reference_path"])
        
        # Get configuration
        if config is None:
            config = ref_module.get_default_config()
        
        # Create inputs
        inputs = ref_module.make_inputs(config)
        
        # Get reference output
        if hasattr(ref_module, "reference"):
            # Call with unpacked inputs if it's a dict
            if isinstance(inputs, dict):
                ref_output = ref_module.reference(**inputs)
            else:
                ref_output = ref_module.reference(inputs)
        else:
            raise AttributeError("Reference module must have a 'reference' function")
        
        # Try to compile/load the generated kernel
        try:
            kernel_module, kernel_fn = load_kernel_module(kernel_source)
            
            if kernel_fn is None:
                result.error_type = "NoEntryPoint"
                result.error_message = "Could not find kernel entry point function"
                return result
            
            result.compiled = True
            
        except Exception as e:
            result.error_type = type(e).__name__
            result.error_message = str(e)
            result.metadata["traceback"] = traceback.format_exc()
            return result
        
        # Check correctness
        if check_correctness:
            try:
                # Call the kernel
                if isinstance(inputs, dict):
                    kernel_output = kernel_fn(**inputs)
                else:
                    kernel_output = kernel_fn(inputs)
                
                # Use the reference's check function if available
                if hasattr(ref_module, "check"):
                    # Get default tolerances from config or use defaults
                    atol = config.get("atol", 1e-2)
                    rtol = config.get("rtol", 1e-2)
                    
                    try:
                        ref_module.check(ref_output, kernel_output, atol=atol, rtol=rtol)
                        result.correctness = True
                    except AssertionError as e:
                        result.correctness = False
                        result.error_message = f"Correctness check failed: {str(e)}"
                else:
                    # Fall back to torch.allclose
                    if isinstance(ref_output, torch.Tensor) and isinstance(kernel_output, torch.Tensor):
                        atol = config.get("atol", 1e-2)
                        rtol = config.get("rtol", 1e-2)
                        result.correctness = torch.allclose(kernel_output, ref_output, atol=atol, rtol=rtol)
                    else:
                        result.correctness = (ref_output == kernel_output)
                
            except Exception as e:
                result.correctness = False
                result.error_type = type(e).__name__
                result.error_message = f"Correctness evaluation error: {str(e)}"
                result.metadata["traceback"] = traceback.format_exc()
                return result
        
        # Measure performance
        if measure_performance and result.correctness:
            try:
                # Warmup
                for _ in range(warmup):
                    if isinstance(inputs, dict):
                        _ = kernel_fn(**inputs)
                    else:
                        _ = kernel_fn(inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Benchmark
                times = []
                for _ in range(repeat):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                    else:
                        start = time.perf_counter()
                    
                    if isinstance(inputs, dict):
                        _ = kernel_fn(**inputs)
                    else:
                        _ = kernel_fn(inputs)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms
                
                result.runtime = sum(times) / len(times)  # Mean time in ms
                result.runtime_stats = {
                    "mean_ms": result.runtime,
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "std_ms": (sum((t - result.runtime) ** 2 for t in times) / len(times)) ** 0.5,
                }
                
                # Also benchmark reference for speedup calculation
                ref_times = []
                for _ in range(repeat):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                    else:
                        start = time.perf_counter()
                    
                    if isinstance(inputs, dict):
                        _ = ref_module.reference(**inputs)
                    else:
                        _ = ref_module.reference(inputs)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    ref_times.append((end - start) * 1000)
                
                ref_mean = sum(ref_times) / len(ref_times)
                result.runtime_stats["reference_mean_ms"] = ref_mean
                result.runtime_stats["speedup"] = ref_mean / result.runtime if result.runtime > 0 else 0.0
                
            except Exception as e:
                result.error_type = type(e).__name__
                result.error_message = f"Performance measurement error: {str(e)}"
                result.metadata["traceback"] = traceback.format_exc()
        
    except Exception as e:
        result.error_type = type(e).__name__
        result.error_message = str(e)
        result.metadata["traceback"] = traceback.format_exc()
    
    return result


def eval_tilebench_kernel_on_suite(
    task_info: Dict,
    kernel_source: str,
    warmup: int = 3,
    repeat: int = 10,
) -> Dict[str, TileBenchEvalResult]:
    """
    Evaluate kernel on all shape configurations in the test suite.
    
    Args:
        task_info: Task information dictionary
        kernel_source: Generated kernel source code
        warmup: Warmup runs per configuration
        repeat: Benchmark repetitions per configuration
        
    Returns:
        Dictionary mapping config description to evaluation results
    """
    # Load reference module to get shape suites
    ref_module = load_reference_module(task_info["reference_path"])
    
    if not hasattr(ref_module, "get_shape_suites"):
        # Fall back to single default config
        return {
            "default": eval_tilebench_kernel(
                task_info, kernel_source, config=None, warmup=warmup, repeat=repeat
            )
        }
    
    shape_suites = ref_module.get_shape_suites()
    results = {}
    
    for i, config in enumerate(shape_suites):
        config_name = f"config_{i+1}"
        if isinstance(config, dict):
            # Create a readable name from key parameters
            key_params = {k: v for k, v in config.items() if k not in ["dtype", "device"]}
            config_name = "_".join(f"{k}{v}" for k, v in key_params.items())
        
        result = eval_tilebench_kernel(
            task_info,
            kernel_source,
            config=config,
            warmup=warmup,
            repeat=repeat,
        )
        results[config_name] = result
    
    return results


if __name__ == "__main__":
    # Test evaluation
    from src.dataset_tilebench import get_task_by_name
    
    print("Testing TileBench Eval")
    print("=" * 50)
    
    try:
        # Get a task
        task_info = get_task_by_name("online_softmax", level="basic")
        print(f"Task: {task_info['task_name']}")
        
        # Load reference
        ref_module = load_reference_module(task_info["reference_path"])
        print(f"Reference loaded: {ref_module.__name__}")
        
        # Test with reference implementation
        config = ref_module.get_default_config()
        print(f"Config: {config}")
        
        inputs = ref_module.make_inputs(config)
        print(f"Inputs: {list(inputs.keys()) if isinstance(inputs, dict) else type(inputs)}")
        
        if isinstance(inputs, dict):
            output = ref_module.reference(**inputs)
        else:
            output = ref_module.reference(inputs)
        print(f"Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

