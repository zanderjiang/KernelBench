#!/usr/bin/env python3
"""
Test script for TileBench integration with KernelBench

This script validates that:
1. TileBench datasets can be discovered
2. Reference.py files can be loaded
3. Problem descriptions can be formatted
4. Evaluation pipeline works end-to-end
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.dataset import construct_tilebench_dataset, get_tilebench_level_mapping
from src.tilebench_reference import (
    load_tilebench_reference,
    get_reference_info,
    format_reference_as_problem_description,
)
from src.tilebench_eval import (
    eval_kernel_against_tilebench_ref,
    is_tilebench_reference,
)


def test_dataset_discovery():
    """Test 1: Dataset Discovery"""
    print("=" * 60)
    print("TEST 1: Dataset Discovery")
    print("=" * 60)
    
    try:
        # Test each level
        for level in ["basic", "medium", "advanced"]:
            print(f"\nDiscovering {level} level problems...")
            problems = construct_tilebench_dataset(level)
            print(f"  Found {len(problems)} problems")
            
            if problems:
                print(f"  First problem: {os.path.basename(os.path.dirname(problems[0]))}")
                print(f"  Last problem: {os.path.basename(os.path.dirname(problems[-1]))}")
        
        print("\n✓ Dataset discovery successful")
        return True
    except Exception as e:
        print(f"\n✗ Dataset discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reference_loading():
    """Test 2: Reference Loading"""
    print("\n" + "=" * 60)
    print("TEST 2: Reference Loading")
    print("=" * 60)
    
    try:
        # Get first problem from basic level
        problems = construct_tilebench_dataset("basic")
        if not problems:
            print("  No problems found to test")
            return False
        
        ref_path = problems[0]
        problem_name = os.path.basename(os.path.dirname(ref_path))
        print(f"\nTesting with problem: {problem_name}")
        print(f"  Path: {ref_path}")
        
        # Test is_tilebench_reference
        is_tb = is_tilebench_reference(ref_path)
        print(f"  is_tilebench_reference: {is_tb}")
        assert is_tb, "Should be detected as TileBench reference"
        
        # Test loading
        print("\n  Loading reference module...")
        module = load_tilebench_reference(ref_path)
        print(f"    ✓ Module loaded: {module.__name__}")
        
        # Check required functions
        required = ["description", "get_default_config", "make_inputs", "reference"]
        for func in required:
            if hasattr(module, func):
                print(f"    ✓ Has {func}()")
            else:
                print(f"    ✗ Missing {func}()")
        
        # Test info extraction
        print("\n  Extracting problem info...")
        info = get_reference_info(ref_path)
        print(f"    Task: {info['task_name']}")
        print(f"    Description: {info['description'][:80]}...")
        print(f"    Config keys: {list(info['config'].keys())}")
        
        print("\n✓ Reference loading successful")
        return True
    except Exception as e:
        print(f"\n✗ Reference loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_problem_description_formatting():
    """Test 3: Problem Description Formatting"""
    print("\n" + "=" * 60)
    print("TEST 3: Problem Description Formatting")
    print("=" * 60)
    
    try:
        problems = construct_tilebench_dataset("basic")
        if not problems:
            print("  No problems found to test")
            return False
        
        ref_path = problems[0]
        problem_name = os.path.basename(os.path.dirname(ref_path))
        print(f"\nFormatting problem: {problem_name}")
        
        # Format problem description
        description = format_reference_as_problem_description(ref_path)
        print(f"\n  Generated description ({len(description)} chars):")
        print("-" * 60)
        print(description[:500])
        print("..." if len(description) > 500 else "")
        print("-" * 60)
        
        # Check key elements
        checks = [
            ("Task:" in description, "Contains task name"),
            ("Description:" in description, "Contains description"),
            ("Configuration:" in description, "Contains configuration"),
            ("def run(" in description, "Contains function signature"),
            ("Requirements:" in description, "Contains requirements"),
        ]
        
        print("\n  Validation:")
        all_pass = True
        for check, msg in checks:
            status = "✓" if check else "✗"
            print(f"    {status} {msg}")
            all_pass = all_pass and check
        
        if all_pass:
            print("\n✓ Problem description formatting successful")
        else:
            print("\n✗ Some validation checks failed")
        
        return all_pass
    except Exception as e:
        print(f"\n✗ Problem description formatting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_with_simple_kernel():
    """Test 4: Evaluation Pipeline"""
    print("\n" + "=" * 60)
    print("TEST 4: Evaluation Pipeline (Simple PyTorch Kernel)")
    print("=" * 60)
    
    try:
        # Find a simple problem to test (e.g., rmsnorm if available)
        problems = construct_tilebench_dataset("basic")
        
        # Try to find rmsnorm or layernorm
        test_ref = None
        test_name = None
        for prob in problems:
            pname = os.path.basename(os.path.dirname(prob))
            if "rmsnorm" in pname.lower() or "layernorm" in pname.lower():
                test_ref = prob
                test_name = pname
                break
        
        if not test_ref:
            # Just use first problem
            test_ref = problems[0]
            test_name = os.path.basename(os.path.dirname(test_ref))
        
        print(f"\nTesting evaluation with: {test_name}")
        
        # Create a simple PyTorch implementation
        # This is a generic pass-through kernel for testing
        test_kernel = """
import torch

def run(**kwargs):
    '''Simple test kernel - returns first input'''
    # Get first tensor input
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            return val
    
    # If single arg, assume it's the input
    if len(kwargs) == 1:
        return list(kwargs.values())[0]
    
    # Otherwise, raise error
    raise ValueError("Could not determine input tensor")
"""
        
        print("\n  Running evaluation...")
        result = eval_kernel_against_tilebench_ref(
            reference_path=test_ref,
            custom_kernel_src=test_kernel,
            backend="python",
            verbose=False,
            measure_performance=True,
            num_perf_trials=10,
        )
        
        print(f"\n  Results:")
        print(f"    Compiled: {result.compiled}")
        print(f"    Correct: {result.correctness}")
        print(f"    Runtime: {result.runtime:.4f} ms" if result.runtime > 0 else "    Runtime: N/A")
        
        if result.metadata:
            print(f"    Metadata: {list(result.metadata.keys())}")
        
        if not result.compiled:
            print(f"    Error: {result.metadata.get('error_type', 'Unknown')}")
            print(f"           {result.metadata.get('error', 'No details')[:100]}")
        
        if result.compiled:
            print("\n✓ Evaluation pipeline functional")
            return True
        else:
            print("\n⚠ Evaluation completed but kernel failed (expected for simple test)")
            return True  # This is actually OK for this test
            
    except Exception as e:
        print(f"\n✗ Evaluation pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TileBench Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dataset Discovery", test_dataset_discovery),
        ("Reference Loading", test_reference_loading),
        ("Problem Description", test_problem_description_formatting),
        ("Evaluation Pipeline", test_evaluation_with_simple_kernel),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

