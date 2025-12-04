"""
Generate and evaluate a single TileBench task

Example usage:
python scripts/generate_and_eval_single_sample_tilebench.py \
    level=basic \
    task_name=online_softmax \
    server_type=openai \
    model_name=gpt-4 \
    prompt_style=full
"""

import os
import sys
import json
import pydra
from pydra import REQUIRED, Config

# Add parent directory to path for imports
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_TOP_DIR)

from src.dataset_tilebench import get_task_by_name, get_task_by_index
from src.prompt_constructor_tilebench import (
    construct_tilebench_prompt_simple,
    add_system_instructions,
)
from src.eval_tilebench import eval_tilebench_kernel, eval_tilebench_kernel_on_suite
from src.utils import (
    create_inference_server_from_presets,
    extract_first_code,
)


class TileBenchEvalConfig(Config):
    def __init__(self):
        # Task specification
        self.level = REQUIRED  # basic, medium, or advanced
        self.task_name = None  # Name of task (e.g., "online_softmax")
        self.task_index = None  # Or use 0-based index instead of name
        
        # Prompt configuration
        self.prompt_style = "full"  # minimal, standard, or full
        self.include_system = True  # Include system instructions
        self.backend = "tilelang"  # Target backend
        
        # Inference configuration
        self.server_type = REQUIRED  # openai, anthropic, google, etc.
        self.model_name = REQUIRED  # Model to use
        self.max_tokens = 8192
        self.temperature = 0.0
        
        # Reasoning model parameters
        self.is_reasoning_model = False
        self.reasoning_effort = None  # low, medium, high
        self.budget_tokens = 0
        
        # Evaluation configuration
        self.warmup = 3
        self.repeat = 20
        self.check_correctness = True
        self.measure_performance = True
        self.eval_on_suite = False  # Evaluate on all shape configurations
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "runs/tilebench")
        self.log = True
        self.verbose = False
        
    def __repr__(self):
        return f"TileBenchEvalConfig({self.to_dict()})"


@pydra.main(base=TileBenchEvalConfig)
def main(config: TileBenchEvalConfig):
    """Generate and evaluate a single TileBench task"""
    
    print("=" * 70)
    print("TileBench Single Task Evaluation")
    print("=" * 70)
    print(f"Config: {config}")
    print()
    
    # 1. Get task information
    print("1. Loading task...")
    try:
        if config.task_name:
            task_info = get_task_by_name(config.task_name, level=config.level)
        elif config.task_index is not None:
            task_info = get_task_by_index(config.level, config.task_index)
        else:
            raise ValueError("Must specify either task_name or task_index")
        
        print(f"   Task: {task_info['task_name']}")
        print(f"   Level: {task_info['level']}")
        print(f"   Reference: {task_info['reference_path']}")
        print(f"   Available prompts: {list(task_info['prompt_files'].keys())}")
        print()
    except Exception as e:
        print(f"   ERROR: Failed to load task: {e}")
        return 1
    
    # 2. Construct prompt
    print("2. Constructing prompt...")
    try:
        prompt = construct_tilebench_prompt_simple(task_info, style=config.prompt_style)
        
        if config.include_system:
            prompt = add_system_instructions(prompt, backend=config.backend)
        
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Style: {config.prompt_style}")
        
        if config.log:
            log_dir = os.path.join(config.logdir, task_info['task_name'])
            os.makedirs(log_dir, exist_ok=True)
            
            prompt_file = os.path.join(log_dir, "prompt.txt")
            with open(prompt_file, "w") as f:
                f.write(prompt)
            print(f"   Saved to: {prompt_file}")
        
        if config.verbose:
            print("\n   --- PROMPT PREVIEW ---")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        print()
    except Exception as e:
        print(f"   ERROR: Failed to construct prompt: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 3. Generate kernel code
    print("3. Generating kernel code...")
    try:
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
        
        # Extract code from response
        kernel_source = extract_first_code(response)
        
        if not kernel_source:
            print("   ERROR: No code found in model response")
            print(f"   Response preview: {response[:500]}")
            return 1
        
        print(f"   Generated {len(kernel_source)} characters of code")
        
        if config.log:
            log_dir = os.path.join(config.logdir, task_info['task_name'])
            
            response_file = os.path.join(log_dir, "response.txt")
            with open(response_file, "w") as f:
                f.write(response)
            
            kernel_file = os.path.join(log_dir, "generated_kernel.py")
            with open(kernel_file, "w") as f:
                f.write(kernel_source)
            
            print(f"   Saved response to: {response_file}")
            print(f"   Saved kernel to: {kernel_file}")
        
        if config.verbose:
            print("\n   --- KERNEL PREVIEW ---")
            print(kernel_source[:500] + "..." if len(kernel_source) > 500 else kernel_source)
        
        print()
    except Exception as e:
        print(f"   ERROR: Failed to generate code: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. Evaluate kernel
    print("4. Evaluating kernel...")
    try:
        if config.eval_on_suite:
            results = eval_tilebench_kernel_on_suite(
                task_info,
                kernel_source,
                warmup=config.warmup,
                repeat=config.repeat,
            )
            
            print(f"   Evaluated on {len(results)} configurations")
            
            all_correct = all(r.correctness for r in results.values())
            all_compiled = all(r.compiled for r in results.values())
            
            print(f"   Compiled: {all_compiled}")
            print(f"   All Correct: {all_correct}")
            
            for config_name, result in results.items():
                print(f"\n   Configuration: {config_name}")
                print(f"     Compiled: {result.compiled}")
                print(f"     Correct: {result.correctness}")
                if result.runtime > 0:
                    print(f"     Runtime: {result.runtime:.4f} ms")
                    if "speedup" in result.runtime_stats:
                        print(f"     Speedup: {result.runtime_stats['speedup']:.2f}x")
                if result.error_message:
                    print(f"     Error: {result.error_message}")
            
            # Save all results
            if config.log:
                log_dir = os.path.join(config.logdir, task_info['task_name'])
                results_file = os.path.join(log_dir, "eval_results.json")
                
                results_dict = {
                    config_name: result.dict()
                    for config_name, result in results.items()
                }
                
                with open(results_file, "w") as f:
                    json.dump(results_dict, f, indent=2)
                
                print(f"\n   Saved results to: {results_file}")
        
        else:
            # Single configuration
            result = eval_tilebench_kernel(
                task_info,
                kernel_source,
                warmup=config.warmup,
                repeat=config.repeat,
                check_correctness=config.check_correctness,
                measure_performance=config.measure_performance,
            )
            
            print(f"   Compiled: {result.compiled}")
            print(f"   Correct: {result.correctness}")
            
            if result.runtime > 0:
                print(f"   Runtime: {result.runtime:.4f} ms")
                if "speedup" in result.runtime_stats:
                    print(f"   Speedup: {result.runtime_stats['speedup']:.2f}x")
                if "reference_mean_ms" in result.runtime_stats:
                    print(f"   Reference: {result.runtime_stats['reference_mean_ms']:.4f} ms")
            
            if result.error_message:
                print(f"   Error: {result.error_message}")
                if config.verbose and "traceback" in result.metadata:
                    print("\n   --- TRACEBACK ---")
                    print(result.metadata["traceback"])
            
            if config.log:
                log_dir = os.path.join(config.logdir, task_info['task_name'])
                results_file = os.path.join(log_dir, "eval_result.json")
                
                with open(results_file, "w") as f:
                    json.dump(result.dict(), f, indent=2)
                
                print(f"\n   Saved result to: {results_file}")
        
        print()
    except Exception as e:
        print(f"   ERROR: Failed to evaluate: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 5. Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Task: {task_info['task_name']} ({task_info['level']})")
    print(f"Model: {config.model_name}")
    
    if config.eval_on_suite:
        print(f"Configurations tested: {len(results)}")
        print(f"All compiled: {all_compiled}")
        print(f"All correct: {all_correct}")
    else:
        print(f"Compiled: {result.compiled}")
        print(f"Correct: {result.correctness}")
        if result.runtime > 0 and "speedup" in result.runtime_stats:
            print(f"Speedup: {result.runtime_stats['speedup']:.2f}x")
    
    if config.log:
        print(f"\nLogs saved to: {os.path.join(config.logdir, task_info['task_name'])}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

