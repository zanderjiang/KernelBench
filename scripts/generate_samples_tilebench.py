"""
Generate kernel samples for multiple TileBench tasks

Example usage:
python scripts/generate_samples_tilebench.py \
    run_name=test_basic \
    level=basic \
    server_type=openai \
    model_name=gpt-4 \
    num_workers=10
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import pydra
from pydra import REQUIRED, Config

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_TOP_DIR)

from src.dataset_tilebench import construct_tilebench_dataset, get_task_info
from src.prompt_constructor_tilebench import (
    construct_tilebench_prompt_simple,
    add_system_instructions,
)
from src.utils import (
    create_inference_server_from_presets,
    extract_first_code,
)


class GenerationConfig(Config):
    def __init__(self):
        # Run configuration
        self.run_name = REQUIRED
        self.level = REQUIRED  # basic, medium, or advanced
        
        # Task selection
        self.tasks = None  # Optional: list of specific task names
        self.skip_tasks = None  # Optional: list of task names to skip
        
        # Prompt configuration
        self.prompt_style = "full"
        self.include_system = True
        self.backend = "tilelang"
        
        # Inference configuration
        self.server_type = REQUIRED
        self.model_name = REQUIRED
        self.max_tokens = 8192
        self.temperature = 0.0
        
        # Reasoning model parameters
        self.is_reasoning_model = False
        self.reasoning_effort = None
        self.budget_tokens = 0
        
        # Parallelization
        self.num_workers = 1
        
        # Output
        self.output_dir = os.path.join(REPO_TOP_DIR, "runs/tilebench")
        self.verbose = False


def generate_for_task(
    task_path: str,
    config: GenerationConfig,
    run_dir: str,
) -> Dict:
    """Generate kernel for a single task"""
    
    task_info = get_task_info(task_path)
    task_name = task_info["task_name"]
    
    print(f"Generating for {task_name}...")
    
    result = {
        "task_name": task_name,
        "level": task_info["level"],
        "success": False,
        "error": None,
    }
    
    try:
        # Construct prompt
        prompt = construct_tilebench_prompt_simple(task_info, style=config.prompt_style)
        
        if config.include_system:
            prompt = add_system_instructions(prompt, backend=config.backend)
        
        # Save prompt
        task_dir = os.path.join(run_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        with open(os.path.join(task_dir, "prompt.txt"), "w") as f:
            f.write(prompt)
        
        # Generate
        inference_server = create_inference_server_from_presets(
            server_type=config.server_type,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            verbose=config.verbose,
            time_generation=True,
            is_reasoning_model=config.is_reasoning_model,
            reasoning_effort=config.reasoning_effort,
            budget_tokens=config.budget_tokens,
        )
        
        response = inference_server(prompt)
        
        # Save response
        with open(os.path.join(task_dir, "response.txt"), "w") as f:
            f.write(response)
        
        # Extract code
        kernel_source = extract_first_code(response)
        
        if kernel_source:
            with open(os.path.join(task_dir, "generated_kernel.py"), "w") as f:
                f.write(kernel_source)
            
            result["success"] = True
            print(f"✓ {task_name}")
        else:
            result["error"] = "No code extracted from response"
            print(f"✗ {task_name}: {result['error']}")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"✗ {task_name}: {result['error']}")
        
        if config.verbose:
            import traceback
            traceback.print_exc()
    
    return result


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """Generate kernels for all tasks in a level"""
    
    print("=" * 70)
    print("TileBench Batch Generation")
    print("=" * 70)
    print(f"Run: {config.run_name}")
    print(f"Level: {config.level}")
    print(f"Model: {config.model_name}")
    print(f"Workers: {config.num_workers}")
    print()
    
    # Get tasks
    all_tasks = construct_tilebench_dataset(config.level)
    
    # Filter tasks if specified
    if config.tasks:
        task_names = set(config.tasks.split(","))
        all_tasks = [t for t in all_tasks if os.path.basename(t) in task_names]
    
    if config.skip_tasks:
        skip_names = set(config.skip_tasks.split(","))
        all_tasks = [t for t in all_tasks if os.path.basename(t) not in skip_names]
    
    print(f"Tasks to process: {len(all_tasks)}")
    for i, task_path in enumerate(all_tasks, 1):
        print(f"  {i}. {os.path.basename(task_path)}")
    print()
    
    # Create run directory
    run_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Generate in parallel
    print("Starting generation...")
    print()
    
    results = []
    
    if config.num_workers == 1:
        # Sequential
        for task_path in all_tasks:
            result = generate_for_task(task_path, config, run_dir)
            results.append(result)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(generate_for_task, task_path, config, run_dir): task_path
                for task_path in all_tasks
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    # Save results
    with open(os.path.join(run_dir, "generation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed tasks:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['task_name']}: {r['error']}")
    
    print(f"\nResults saved to: {run_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

