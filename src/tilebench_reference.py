"""
Simple loader for TileBench reference.py files

Loads reference module and extracts:
- description(): Task description
- get_default_config(): Input specs
- make_inputs(cfg): Create test inputs
- reference(**inputs): Ground truth
- check(y_ref, y_out, atol, rtol): Correctness check
"""

import importlib.util
import os
import sys
from typing import Any, Dict, Optional


def load_tilebench_reference(reference_path: str):
    """
    Load a TileBench reference.py module
    
    Args:
        reference_path: Path to reference.py file
        
    Returns:
        Loaded module with standard TileBench API
    """
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference not found: {reference_path}")
    
    # Create unique module name
    task_name = os.path.basename(os.path.dirname(reference_path))
    module_name = f"tilebench_ref_{task_name}"
    
    # Load module
    spec = importlib.util.spec_from_file_location(module_name, reference_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def get_reference_info(reference_path: str) -> Dict[str, Any]:
    """
    Extract key information from a TileBench reference for prompting
    
    Args:
        reference_path: Path to reference.py
        
    Returns:
        Dict with:
        - description: Task description
        - config: Default config dict
        - input_signature: String describing inputs
        - task_name: Name of the task
    """
    module = load_tilebench_reference(reference_path)
    
    # Get description
    description = module.description() if hasattr(module, "description") else "No description"
    
    # Get default config
    config = module.get_default_config()
    
    # Extract input signature from make_inputs
    import inspect
    import torch
    if hasattr(module, "make_inputs"):
        # Create sample inputs to see structure
        sample_inputs = module.make_inputs(config)
        
        if isinstance(sample_inputs, dict):
            sig_parts = []
            for k, v in sample_inputs.items():
                if isinstance(v, torch.Tensor):
                    sig_parts.append(f"{k}: Tensor{list(v.shape)}")
                else:
                    # Scalar parameter
                    sig_parts.append(f"{k}: {type(v).__name__}")
            input_signature = ", ".join(sig_parts)
        else:
            if isinstance(sample_inputs, torch.Tensor):
                input_signature = f"input: Tensor{list(sample_inputs.shape)}"
            else:
                input_signature = f"input: {type(sample_inputs).__name__}"
    else:
        input_signature = "Unknown"
    
    # Task name
    task_name = os.path.basename(os.path.dirname(reference_path))
    
    return {
        "description": description,
        "config": config,
        "input_signature": input_signature,
        "task_name": task_name,
        "reference_path": reference_path,
    }


def format_reference_as_problem_description(reference_path: str) -> str:
    """
    Format TileBench reference as a problem description for KernelBench prompts
    
    This creates a "reference architecture" string that KernelBench's
    prompt constructor can use.
    
    Args:
        reference_path: Path to reference.py
        
    Returns:
        Formatted problem description string
    """
    info = get_reference_info(reference_path)
    
    # Format config
    config_str = "\n".join(f"# {k}: {v}" for k, v in info["config"].items())
    
    # Format as simple code with clear instructions
    # The key instruction is embedded as a comment
    problem_description = f'''# Task: {info["task_name"]}
# Description: {info["description"]}
# 
{config_str}

# IMPORTANT: Implement a standalone 'run()' function
# Your output should be ONLY a function definition.

def run({info["input_signature"]}):
    """
    {info["description"]}
    
    Args: {info["input_signature"]}
    Returns: output tensor
    
    Use torch.utils.cpp_extension.load_inline to compile CUDA kernels inline.
    Must match reference behavior within tolerance (atol=1e-2, rtol=1e-2).
    """
    # TODO: Implement using load_inline with CUDA kernels
    # Your implementation here
    pass
'''
    
    return problem_description


# List of all TileBench tasks
def discover_tilebench_tasks(tilebench_benchmark_path: str, level: str = "basic"):
    """
    Discover all TileBench tasks in a level
    
    Args:
        tilebench_benchmark_path: Path to TileBench-Benchmark/
        level: "basic", "medium", or "advanced"
        
    Returns:
        List of (task_name, reference_path) tuples
    """
    level_dir = os.path.join(tilebench_benchmark_path, level)
    
    if not os.path.exists(level_dir):
        raise ValueError(f"Level directory not found: {level_dir}")
    
    tasks = []
    for task_name in sorted(os.listdir(level_dir)):
        task_path = os.path.join(level_dir, task_name)
        
        if not os.path.isdir(task_path):
            continue
        
        if task_name.startswith("example-"):
            continue
        
        reference_path = os.path.join(task_path, "reference.py")
        if os.path.exists(reference_path):
            tasks.append((task_name, reference_path))
    
    return tasks


if __name__ == "__main__":
    # Test
    import sys
    
    if len(sys.argv) > 1:
        ref_path = sys.argv[1]
    else:
        # Default test
        ref_path = "../../TileBench-Benchmark/basic/rmsnorm/reference.py"
    
    print(f"Testing with: {ref_path}")
    print("=" * 60)
    
    # Test loading
    module = load_tilebench_reference(ref_path)
    print(f"✓ Loaded module: {module.__name__}")
    
    # Test info extraction
    info = get_reference_info(ref_path)
    print(f"\n✓ Extracted info:")
    print(f"  Task: {info['task_name']}")
    print(f"  Description: {info['description']}")
    print(f"  Config: {info['config']}")
    print(f"  Input signature: {info['input_signature']}")
    
    # Test problem description formatting
    problem_desc = format_reference_as_problem_description(ref_path)
    print(f"\n✓ Formatted problem description ({len(problem_desc)} chars)")
    print("\n" + problem_desc)

