import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
import os
from .benchmark_toy import benchmark,run_operation1,run_operation2
from .triton_gelu import triton_gelu
def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")

def ensure_directory_exists(path: str):
    """Create directory at `path` if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def create_cuda_gelu():
    cuda_gelu_src = open("./toy_things/gelu.cu").read()
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    ensure_directory_exists("./toy_things/var/cuda_gelu")
    if not torch.cuda.is_available():
        return None
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="./toy_things/var/cuda_gelu",
        )
    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu

if __name__ == "__main__":
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu()))
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))