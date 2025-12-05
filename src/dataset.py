################################################################################
# Helpers for Dataset
################################################################################

import os
import random
import re
import hashlib

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
TILEBENCH_BENCHMARK_PATH = os.path.join(REPO_TOP_PATH, "..", "..", "TileBench-Benchmark")


def assign_problem_hash(problem_path: str) -> list[int]:
    """
    Assign a unique hash to a problem in the dataset
    """
    with open(problem_path, "r") as f:
        problem_src = f.read()
    return get_code_hash(problem_src)


def get_code_hash(problem_src: str) -> str:
    """
    Assign a unique hash to some piece of code
    Important to strip out the comments and whitespace as they are not functionally part of the code
    """
    # Remove multi-line comments first
    problem_src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", problem_src)
    # Remove inline comments and all whitespace
    cleaned_problem_src = re.sub(r"#.*$|\s+", "", problem_src, flags=re.MULTILINE)
    # hash only on code
    return hashlib.md5(cleaned_problem_src.encode()).hexdigest()


def construct_problem_dataset_from_problem_dir(problem_dir: str) -> list[str]:
    """
    Construct a list of relative paths to all the python files in the problem directory
    Sorted by the numerical prefix of the filenames
    """
    DATASET = []

    for file_name in os.listdir(problem_dir):
        if file_name.endswith(".py"):
            # TODO: revisit later to satisfy eval harnes
            relative_path = os.path.join(problem_dir, file_name)
            DATASET.append(relative_path)

    # Sort the DATASET based on the numerical prefix of the filenames
    DATASET.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))

    return DATASET


def construct_kernelbench_dataset(level: int) -> list[str]:
    return construct_problem_dataset_from_problem_dir(
        os.path.join(KERNEL_BENCH_PATH, f"level{level}")
    )


KERNELBENCH_LEVEL_1_DATASET = construct_kernelbench_dataset(level=1)
KERNELBENCH_LEVEL_2_DATASET = construct_kernelbench_dataset(level=2)
KERNELBENCH_LEVEL_3_DATASET = construct_kernelbench_dataset(level=3)

################################################################################
# Eval on Subsets of KernelBench
################################################################################


def get_kernelbench_subset(
    level: int, num_subset_problems: int = 10, random_seed: int = 42
) -> tuple[list[str], list[int]]:
    """
    Get a random subset of problems from the KernelBench dataset
    """

    full_dataset = construct_kernelbench_dataset(level)

    random.seed(random_seed)
    num_subset_problems = min(num_subset_problems, len(full_dataset))
    subset_indices = random.sample(range(len(full_dataset)), num_subset_problems)

    subset = sorted([full_dataset[i] for i in subset_indices])
    return subset, subset_indices


################################################################################
# Representative subsets of KernelBench
# use this if you want to iterate on methods without the hassle of running the full dataset
# problem_ids are 1-indexed (logical index)
################################################################################

level1_representative_subset = [
    "1_Square_matrix_multiplication_.py",
    "3_Batched_matrix_multiplication.py",
    "6_Matmul_with_large_K_dimension_.py",
    "18_Matmul_with_transposed_both.py",
    "23_Softmax.py",
    "26_GELU_.py",
    "33_BatchNorm.py",
    "36_RMSNorm_.py",
    "40_LayerNorm.py",
    "42_Max_Pooling_2D.py",
    "48_Mean_reduction_over_a_dimension.py",
    "54_conv_standard_3D__square_input__square_kernel.py",
    "57_conv_transposed_2D__square_input__square_kernel.py",
    "65_conv_transposed_2D__square_input__asymmetric_kernel.py",
    "77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__.py",
    "82_conv_depthwise_2D_square_input_square_kernel.py",
    "86_conv_depthwise_separable_2D.py",
    "87_conv_pointwise_2D.py",
]

level1_representative_subset_problem_ids = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]

level2_representative_subset = [
    "1_Conv2D_ReLU_BiasAdd.py",
    "2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py",
    "8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py",
    "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",
    "23_Conv3d_GroupNorm_Mean.py",
    "28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py",
    "33_Gemm_Scale_BatchNorm.py",
    "43_Conv3d_Max_LogSumExp_ReLU.py",
]

level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]

level3_representative_subset = [
    "1_MLP.py",
    "5_AlexNet.py",
    "8_ResNetBasicBlock.py",
    "11_VGG16.py",
    "20_MobileNetV2.py",
    "21_EfficientNetMBConv.py",
    "33_VanillaRNN.py",
    "38_LTSMBidirectional.py",
    "43_MinGPTCausalAttention.py",
]

level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]


################################################################################
# TileBench Dataset Construction
################################################################################

def construct_tilebench_dataset(level: str) -> list[str]:
    """
    Construct a list of paths to TileBench reference.py files for a given level.
    
    Args:
        level: "basic", "medium", or "advanced"
        
    Returns:
        List of paths to reference.py files, sorted alphabetically by problem name
    """
    level_dir = os.path.join(TILEBENCH_BENCHMARK_PATH, level)
    
    if not os.path.exists(level_dir):
        raise ValueError(f"TileBench level directory not found: {level_dir}")
    
    dataset = []
    
    # Iterate through problem directories
    for problem_name in sorted(os.listdir(level_dir)):
        problem_path = os.path.join(level_dir, problem_name)
        
        # Skip if not a directory
        if not os.path.isdir(problem_path):
            continue
        
        # Skip example problems
        if problem_name.startswith("example-"):
            continue
        
        # Check if reference.py exists
        reference_path = os.path.join(problem_path, "reference.py")
        if os.path.exists(reference_path):
            dataset.append(reference_path)
    
    return dataset


def get_tilebench_level_mapping() -> dict:
    """
    Map numeric levels to TileBench level names for backward compatibility.
    
    Returns:
        Dict mapping integers to level names
    """
    return {
        1: "basic",
        2: "medium",
        3: "advanced",
    }


def construct_dataset_unified(level) -> list[str]:
    """
    Unified dataset constructor that works with both KernelBench and TileBench.
    
    Args:
        level: Either an integer (1-3 for TileBench, 1-4 for KernelBench) or 
               a string ("basic", "medium", "advanced")
    
    Returns:
        List of paths to problem files (reference.py for TileBench, .py for KernelBench)
    """
    # If level is a string, assume TileBench
    if isinstance(level, str):
        return construct_tilebench_dataset(level)
    
    # If level is an integer, check if TileBench path exists
    if os.path.exists(TILEBENCH_BENCHMARK_PATH):
        level_mapping = get_tilebench_level_mapping()
        if level in level_mapping:
            return construct_tilebench_dataset(level_mapping[level])
    
    # Fall back to KernelBench
    return construct_kernelbench_dataset(level)