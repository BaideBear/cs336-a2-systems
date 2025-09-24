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
from .triton_softmax import triton_softmax

def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_equal2(f1, f2):
    x = torch.randn(2048, 2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)

def manual_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y

def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


if __name__ == "__main__":
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=get_device())
    y1 = manual_softmax(x) # @inspect y1
    y2 = triton_softmax(x)
    assert torch.allclose(y1, y2)

    check_equal2(pytorch_softmax, manual_softmax)
    check_equal2(pytorch_softmax, triton_softmax)

    compiled_softmax = torch.compile(manual_softmax)

    manual_time = benchmark("manual_softmax", run_operation1(dim=16384, operation=manual_softmax)) # @inspect manual_time
    compiled_time = benchmark("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax)) # @inspect compiled_time
    pytorch_time = benchmark("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax)) # @inspect pytorch_time
    triton_time = benchmark("triton_softmax", run_operation1(dim=16384, operation=triton_softmax)) # @inspect triton_time


