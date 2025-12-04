"""
Prompt constructor for TileBench tasks

TileBench tasks come with their own prompt files:
- prompt_codegen.md: Core code generation instructions
- prompt_fewshot.md: Few-shot examples (optional)
- prompt_correctness.md: Correctness checking guidelines (optional)
- prompt_performance.md: Performance optimization tips (optional)
- prompt_precautions.md: Common pitfalls (optional)
- prompt_task_examples_notes.md: Task-specific examples (optional)
- prompt.md: Main combined prompt (optional)
"""

import os
from typing import Dict, List, Optional
from src.utils import read_file


def read_prompt_file(prompt_path: str) -> str:
    """Read and return contents of a prompt file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return read_file(prompt_path).strip()


def construct_tilebench_prompt(
    task_info: Dict,
    prompt_components: Optional[List[str]] = None,
    include_fewshot: bool = True,
    include_correctness: bool = True,
    include_performance: bool = True,
    include_precautions: bool = True,
    include_examples: bool = False,
) -> str:
    """
    Construct a prompt for a TileBench task from its prompt files.
    
    Args:
        task_info: Task information dictionary from dataset_tilebench.get_task_info()
        prompt_components: Optional list of specific components to include in order.
                          If None, uses default ordering.
                          Valid components: "codegen", "fewshot", "correctness", 
                                           "performance", "precautions", "task_examples_notes"
        include_fewshot: Whether to include few-shot examples (default: True)
        include_correctness: Whether to include correctness guidelines (default: True)
        include_performance: Whether to include performance tips (default: True)
        include_precautions: Whether to include precautions (default: True)
        include_examples: Whether to include task examples (default: False)
        
    Returns:
        Complete prompt string
    """
    prompt_files = task_info.get("prompt_files", {})
    
    # If prompt.md exists, use it as the complete prompt
    if "main" in prompt_files:
        return read_prompt_file(prompt_files["main"])
    
    # Otherwise, construct from components
    if prompt_components is None:
        # Default ordering
        prompt_components = []
        
        # Core codegen always comes first
        if "codegen" in prompt_files:
            prompt_components.append("codegen")
        
        # Optional components
        if include_fewshot and "fewshot" in prompt_files:
            prompt_components.append("fewshot")
        
        if include_examples and "task_examples_notes" in prompt_files:
            prompt_components.append("task_examples_notes")
            
        if include_correctness and "correctness" in prompt_files:
            prompt_components.append("correctness")
        
        if include_precautions and "precautions" in prompt_files:
            prompt_components.append("precautions")
            
        if include_performance and "performance" in prompt_files:
            prompt_components.append("performance")
    
    # Build the prompt from components
    prompt_parts = []
    
    for component in prompt_components:
        if component in prompt_files:
            content = read_prompt_file(prompt_files[component])
            prompt_parts.append(content)
        else:
            print(f"Warning: Prompt component '{component}' not found for task '{task_info['task_name']}'")
    
    if not prompt_parts:
        raise ValueError(f"No prompt components found for task '{task_info['task_name']}'")
    
    # Join with double newlines for clear separation
    return "\n\n".join(prompt_parts)


def construct_tilebench_prompt_simple(
    task_info: Dict,
    style: str = "full"
) -> str:
    """
    Simplified prompt construction with preset styles.
    
    Args:
        task_info: Task information dictionary
        style: Prompt style - one of:
               - "minimal": Just codegen instructions
               - "standard": Codegen + fewshot + correctness
               - "full": All available components (default)
               
    Returns:
        Complete prompt string
    """
    if style == "minimal":
        return construct_tilebench_prompt(
            task_info,
            include_fewshot=False,
            include_correctness=False,
            include_performance=False,
            include_precautions=False,
            include_examples=False,
        )
    elif style == "standard":
        return construct_tilebench_prompt(
            task_info,
            include_fewshot=True,
            include_correctness=True,
            include_performance=False,
            include_precautions=False,
            include_examples=False,
        )
    elif style == "full":
        return construct_tilebench_prompt(
            task_info,
            include_fewshot=True,
            include_correctness=True,
            include_performance=True,
            include_precautions=True,
            include_examples=True,
        )
    else:
        raise ValueError(f"Unknown prompt style: {style}. Choose from: minimal, standard, full")


def get_tilebench_reference_source(task_info: Dict) -> str:
    """
    Get the reference.py source code for a TileBench task.
    
    Args:
        task_info: Task information dictionary
        
    Returns:
        Source code of reference.py as string
    """
    reference_path = task_info["reference_path"]
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    
    return read_file(reference_path)


def construct_prompt_with_reference(
    task_info: Dict,
    style: str = "full",
    include_reference: bool = False,
) -> str:
    """
    Construct a complete prompt, optionally including the reference implementation.
    
    Args:
        task_info: Task information dictionary
        style: Prompt style (minimal/standard/full)
        include_reference: Whether to include reference.py in the prompt
        
    Returns:
        Complete prompt string
    """
    prompt = construct_tilebench_prompt_simple(task_info, style=style)
    
    if include_reference:
        reference_code = get_tilebench_reference_source(task_info)
        prompt += f"\n\n## Reference Implementation\n\n```python\n{reference_code}\n```\n"
    
    return prompt


def add_system_instructions(prompt: str, backend: str = "tilelang") -> str:
    """
    Add system-level instructions for code generation.
    
    Args:
        prompt: Base prompt
        backend: Target backend (default: tilelang)
        
    Returns:
        Prompt with system instructions prepended
    """
    system_instructions = f"""You are an expert GPU kernel developer specializing in {backend}.

Your task is to implement a high-performance kernel based on the specifications provided.

Key requirements:
1. The code must be complete and runnable
2. It must match the reference implementation's behavior
3. Optimize for performance on modern GPUs
4. Follow best practices for {backend}

Generate ONLY the code implementation. Do not include explanations or markdown formatting unless specifically requested.

---

"""
    
    return system_instructions + prompt


def log_prompt(prompt: str, output_dir: str, task_name: str):
    """
    Save prompt to a file for inspection.
    
    Args:
        prompt: The prompt text
        output_dir: Directory to save to
        task_name: Name of the task
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{task_name}_prompt.txt")
    
    with open(output_path, "w") as f:
        f.write(prompt)
    
    print(f"Prompt saved to: {output_path}")


if __name__ == "__main__":
    # Test prompt construction
    from src.dataset_tilebench import get_task_by_name
    
    print("Testing TileBench Prompt Constructor")
    print("=" * 50)
    
    # Test with a basic task
    try:
        task_info = get_task_by_name("online_softmax", level="basic")
        
        print(f"\nTask: {task_info['task_name']}")
        print(f"Level: {task_info['level']}")
        print(f"Available prompts: {list(task_info['prompt_files'].keys())}")
        
        print("\n--- MINIMAL PROMPT ---")
        minimal = construct_tilebench_prompt_simple(task_info, style="minimal")
        print(f"Length: {len(minimal)} chars")
        print(minimal[:500] + "..." if len(minimal) > 500 else minimal)
        
        print("\n--- STANDARD PROMPT ---")
        standard = construct_tilebench_prompt_simple(task_info, style="standard")
        print(f"Length: {len(standard)} chars")
        
        print("\n--- FULL PROMPT ---")
        full = construct_tilebench_prompt_simple(task_info, style="full")
        print(f"Length: {len(full)} chars")
        
        # Test with system instructions
        with_system = add_system_instructions(minimal, backend="tilelang")
        print(f"\n--- WITH SYSTEM INSTRUCTIONS ---")
        print(f"Length: {len(with_system)} chars")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

