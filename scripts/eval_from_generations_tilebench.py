"""
Evaluate generated TileBench kernels from a run directory

Example usage:
python scripts/eval_from_generations_tilebench.py \
    run_name=test_basic \
    level=basic \
    num_workers=4
"""

import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pydra
from pydra import REQUIRED, Config

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_TOP_DIR)

from src.dataset_tilebench import get_task_by_name
from src.eval_tilebench import eval_tilebench_kernel, eval_tilebench_kernel_on_suite
from src.utils import read_file


class EvalConfig(Config):
    def __init__(self):
        # Run configuration
        self.run_name = REQUIRED
        self.level = REQUIRED
        
        # Evaluation settings
        self.warmup = 3
        self.repeat = 20
        self.eval_on_suite = False
        
        # Parallelization
        self.num_workers = 1
        
        # Paths
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs/tilebench")
        self.verbose = False


def eval_task(
    task_name: str,
    level: str,
    run_dir: str,
    config: EvalConfig,
) -> Dict:
    """Evaluate a single generated kernel"""
    
    print(f"Evaluating {task_name}...")
    
    result = {
        "task_name": task_name,
        "level": level,
        "evaluated": False,
        "error": None,
    }
    
    try:
        # Get task info
        task_info = get_task_by_name(task_name, level=level)
        
        # Load generated kernel
        task_dir = os.path.join(run_dir, task_name)
        kernel_path = os.path.join(task_dir, "generated_kernel.py")
        
        if not os.path.exists(kernel_path):
            result["error"] = "No generated kernel found"
            print(f"✗ {task_name}: {result['error']}")
            return result
        
        kernel_source = read_file(kernel_path)
        
        # Evaluate
        if config.eval_on_suite:
            eval_results = eval_tilebench_kernel_on_suite(
                task_info,
                kernel_source,
                warmup=config.warmup,
                repeat=config.repeat,
            )
            
            # Save detailed results
            with open(os.path.join(task_dir, "eval_results.json"), "w") as f:
                results_dict = {
                    config_name: res.dict()
                    for config_name, res in eval_results.items()
                }
                json.dump(results_dict, f, indent=2)
            
            # Aggregate
            result["evaluated"] = True
            result["num_configs"] = len(eval_results)
            result["all_compiled"] = all(r.compiled for r in eval_results.values())
            result["all_correct"] = all(r.correctness for r in eval_results.values())
            result["configs"] = {
                name: {
                    "compiled": r.compiled,
                    "correct": r.correctness,
                    "runtime_ms": r.runtime,
                    "speedup": r.runtime_stats.get("speedup", 0.0),
                }
                for name, r in eval_results.items()
            }
            
            status = "✓" if result["all_correct"] else "✗"
            print(f"{status} {task_name}: {result['num_configs']} configs, "
                  f"correct: {result['all_correct']}")
        
        else:
            eval_result = eval_tilebench_kernel(
                task_info,
                kernel_source,
                warmup=config.warmup,
                repeat=config.repeat,
            )
            
            # Save result
            with open(os.path.join(task_dir, "eval_result.json"), "w") as f:
                json.dump(eval_result.dict(), f, indent=2)
            
            result["evaluated"] = True
            result["compiled"] = eval_result.compiled
            result["correct"] = eval_result.correctness
            result["runtime_ms"] = eval_result.runtime
            result["speedup"] = eval_result.runtime_stats.get("speedup", 0.0)
            
            if eval_result.error_message:
                result["error"] = eval_result.error_message
            
            status = "✓" if eval_result.correctness else "✗"
            speedup_str = f", {result['speedup']:.2f}x" if result["speedup"] > 0 else ""
            print(f"{status} {task_name}{speedup_str}")
    
    except Exception as e:
        result["error"] = str(e)
        print(f"✗ {task_name}: {result['error']}")
        
        if config.verbose:
            import traceback
            traceback.print_exc()
    
    return result


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """Evaluate all generated kernels in a run"""
    
    print("=" * 70)
    print("TileBench Batch Evaluation")
    print("=" * 70)
    print(f"Run: {config.run_name}")
    print(f"Level: {config.level}")
    print()
    
    # Get run directory
    run_dir = os.path.join(config.runs_dir, config.run_name)
    
    if not os.path.exists(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        return 1
    
    # Find all task directories with generated kernels
    task_dirs = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path):
            kernel_path = os.path.join(item_path, "generated_kernel.py")
            if os.path.exists(kernel_path):
                task_dirs.append(item)
    
    print(f"Tasks to evaluate: {len(task_dirs)}")
    for i, task_name in enumerate(sorted(task_dirs), 1):
        print(f"  {i}. {task_name}")
    print()
    
    # Evaluate
    print("Starting evaluation...")
    print()
    
    results = []
    
    if config.num_workers == 1:
        # Sequential
        for task_name in sorted(task_dirs):
            result = eval_task(task_name, config.level, run_dir, config)
            results.append(result)
    else:
        # Parallel
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(eval_task, task_name, config.level, run_dir, config): task_name
                for task_name in task_dirs
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    # Save results
    with open(os.path.join(run_dir, "eval_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Compute statistics
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    total = len(results)
    evaluated = sum(1 for r in results if r["evaluated"])
    
    if config.eval_on_suite:
        all_compiled = sum(1 for r in results if r.get("all_compiled", False))
        all_correct = sum(1 for r in results if r.get("all_correct", False))
        
        print(f"Total tasks: {total}")
        print(f"Evaluated: {evaluated}")
        print(f"All configs compiled: {all_compiled} ({100*all_compiled/total:.1f}%)")
        print(f"All configs correct: {all_correct} ({100*all_correct/total:.1f}%)")
    else:
        compiled = sum(1 for r in results if r.get("compiled", False))
        correct = sum(1 for r in results if r.get("correct", False))
        
        # Calculate speedup stats
        speedups = [r["speedup"] for r in results if r.get("speedup", 0) > 0]
        
        print(f"Total tasks: {total}")
        print(f"Evaluated: {evaluated}")
        print(f"Compiled: {compiled} ({100*compiled/total:.1f}%)")
        print(f"Correct: {correct} ({100*correct/total:.1f}%)")
        
        if speedups:
            print(f"\nSpeedup statistics:")
            print(f"  Mean: {sum(speedups)/len(speedups):.2f}x")
            print(f"  Min: {min(speedups):.2f}x")
            print(f"  Max: {max(speedups):.2f}x")
            print(f"  Median: {sorted(speedups)[len(speedups)//2]:.2f}x")
        
        # fast_p metrics (similar to KernelBench)
        for threshold in [0.0, 1.0, 2.0]:
            fast_p = sum(1 for r in results 
                        if r.get("correct", False) and r.get("speedup", 0) >= threshold)
            print(f"  fast_{threshold:.0f} (≥{threshold}x speedup): "
                  f"{fast_p}/{total} ({100*fast_p/total:.1f}%)")
    
    # List failed tasks
    failed = [r for r in results if not r.get("correct" if not config.eval_on_suite else "all_correct", False)]
    if failed:
        print(f"\nFailed tasks ({len(failed)}):")
        for r in failed:
            error_msg = r.get("error", "incorrect output")
            print(f"  - {r['task_name']}: {error_msg}")
    
    print(f"\nResults saved to: {run_dir}/eval_summary.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

