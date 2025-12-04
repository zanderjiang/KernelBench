"""
Dataset helpers for TileBench

TileBench structure:
- TileBench-Benchmark/
  - basic/
    - task_name/
      - reference.py
      - prompt_codegen.md
      - prompt_fewshot.md (optional)
      - ...
  - medium/
  - advanced/
"""

import os
from typing import List, Tuple, Dict, Any

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
    )
)
TILEBENCH_BENCHMARK_PATH = os.path.join(REPO_TOP_PATH, "TileBench-Benchmark")


def get_tilebench_levels() -> List[str]:
    """Get all difficulty levels in TileBench"""
    return ["basic", "medium", "advanced"]


def construct_tilebench_dataset(level: str) -> List[str]:
    """
    Construct a list of paths to task directories for a given level.
    
    Args:
        level: One of "basic", "medium", "advanced"
        
    Returns:
        List of absolute paths to task directories, sorted alphabetically
    """
    level_dir = os.path.join(TILEBENCH_BENCHMARK_PATH, level)
    
    if not os.path.exists(level_dir):
        raise ValueError(f"Level directory does not exist: {level_dir}")
    
    tasks = []
    for task_name in os.listdir(level_dir):
        task_path = os.path.join(level_dir, task_name)
        
        # Skip if not a directory
        if not os.path.isdir(task_path):
            continue
            
        # Skip example/experiment directories
        if task_name.startswith("example-") or task_name.startswith("experiment-"):
            continue
            
        # Check if reference.py exists
        reference_path = os.path.join(task_path, "reference.py")
        if os.path.exists(reference_path):
            tasks.append(task_path)
    
    # Sort alphabetically by task name
    tasks.sort(key=lambda x: os.path.basename(x))
    
    return tasks


def get_task_info(task_path: str) -> Dict[str, Any]:
    """
    Extract information about a task from its directory.
    
    Args:
        task_path: Path to the task directory
        
    Returns:
        Dictionary with task information:
        - task_name: Name of the task
        - level: Difficulty level (basic/medium/advanced)
        - reference_path: Path to reference.py
        - prompt_files: Dict of available prompt files
        - has_kernel: Whether a kernel.py exists (solution)
    """
    task_name = os.path.basename(task_path)
    
    # Determine level by checking parent directory name
    level = os.path.basename(os.path.dirname(task_path))
    
    reference_path = os.path.join(task_path, "reference.py")
    
    # Check for available prompt files
    prompt_files = {}
    prompt_types = [
        "prompt_codegen.md",
        "prompt_fewshot.md", 
        "prompt_correctness.md",
        "prompt_performance.md",
        "prompt_precautions.md",
        "prompt_task_examples_notes.md",
        "prompt.md",
    ]
    
    for prompt_type in prompt_types:
        prompt_path = os.path.join(task_path, prompt_type)
        if os.path.exists(prompt_path):
            # Extract key from filename (e.g., "codegen" from "prompt_codegen.md")
            key = prompt_type.replace("prompt_", "").replace(".md", "")
            if key == "prompt":  # Handle the base "prompt.md" case
                key = "main"
            prompt_files[key] = prompt_path
    
    # Check for existing kernel solution
    kernel_path = os.path.join(task_path, "kernel.py")
    has_kernel = os.path.exists(kernel_path)
    
    return {
        "task_name": task_name,
        "level": level,
        "reference_path": reference_path,
        "prompt_files": prompt_files,
        "has_kernel": has_kernel,
        "kernel_path": kernel_path if has_kernel else None,
    }


def list_all_tilebench_tasks() -> List[Dict[str, Any]]:
    """
    Get information about all TileBench tasks across all levels.
    
    Returns:
        List of task info dictionaries
    """
    all_tasks = []
    
    for level in get_tilebench_levels():
        try:
            tasks = construct_tilebench_dataset(level)
            for task_path in tasks:
                task_info = get_task_info(task_path)
                all_tasks.append(task_info)
        except Exception as e:
            print(f"Warning: Could not load tasks from level {level}: {e}")
    
    return all_tasks


def get_task_by_name(task_name: str, level: str = None) -> Dict[str, Any]:
    """
    Get task information by task name.
    
    Args:
        task_name: Name of the task (e.g., "online_softmax")
        level: Optional level to narrow search ("basic", "medium", "advanced")
        
    Returns:
        Task info dictionary
        
    Raises:
        ValueError: If task not found
    """
    levels_to_search = [level] if level else get_tilebench_levels()
    
    for search_level in levels_to_search:
        try:
            tasks = construct_tilebench_dataset(search_level)
            for task_path in tasks:
                if os.path.basename(task_path) == task_name:
                    return get_task_info(task_path)
        except Exception:
            continue
    
    raise ValueError(f"Task '{task_name}' not found" + (f" in level '{level}'" if level else ""))


def get_task_by_index(level: str, index: int) -> Dict[str, Any]:
    """
    Get task information by level and index.
    
    Args:
        level: Difficulty level ("basic", "medium", "advanced")
        index: 0-based index in the sorted task list
        
    Returns:
        Task info dictionary
    """
    tasks = construct_tilebench_dataset(level)
    
    if index < 0 or index >= len(tasks):
        raise IndexError(f"Task index {index} out of range for level '{level}' (0-{len(tasks)-1})")
    
    return get_task_info(tasks[index])


# Create dataset constants for each level
TILEBENCH_BASIC_DATASET = construct_tilebench_dataset("basic") if os.path.exists(
    os.path.join(TILEBENCH_BENCHMARK_PATH, "basic")
) else []

TILEBENCH_MEDIUM_DATASET = construct_tilebench_dataset("medium") if os.path.exists(
    os.path.join(TILEBENCH_BENCHMARK_PATH, "medium")
) else []

TILEBENCH_ADVANCED_DATASET = construct_tilebench_dataset("advanced") if os.path.exists(
    os.path.join(TILEBENCH_BENCHMARK_PATH, "advanced")
) else []


if __name__ == "__main__":
    # Test the dataset construction
    print("TileBench Dataset Summary")
    print("=" * 50)
    
    for level in get_tilebench_levels():
        try:
            tasks = construct_tilebench_dataset(level)
            print(f"\n{level.upper()}: {len(tasks)} tasks")
            for i, task_path in enumerate(tasks):
                task_info = get_task_info(task_path)
                print(f"  {i+1}. {task_info['task_name']}")
                print(f"     - Reference: {os.path.exists(task_info['reference_path'])}")
                print(f"     - Prompts: {', '.join(task_info['prompt_files'].keys())}")
                print(f"     - Has solution: {task_info['has_kernel']}")
        except Exception as e:
            print(f"  Error loading {level}: {e}")

