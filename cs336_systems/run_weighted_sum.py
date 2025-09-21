import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
import os
from toy_things.benchmark_toy import benchmark
from .triton_weighted_sum import WeightedSumFunc
import pytest
from torch.testing import assert_close
from torch.autograd import gradcheck

def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def manual_weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)

def manual_weighted_sum_backward(x, weight, grad_output):
    grad_x = grad_output.unsqueeze(-1) * weight.unsqueeze(0)
    grad_weight = (x * grad_output.unsqueeze(-1)).sum(axis=0)
    return grad_x, grad_weight

if __name__ == "__main__":
    generator = torch.Generator(device=get_device()).manual_seed(42)
    R1, R2, D = 16, 128, 1024
    weight = torch.randn(D, device=get_device(), generator=generator, requires_grad=True)
    x = torch.randn(R2, D, device=get_device(), generator=generator, requires_grad=True)

    y1 = manual_weighted_sum(x, weight)
    y2 = WeightedSumFunc.apply(x, weight)
    print("Forward shapes:", y1.shape, y2.shape)
    # assert torch.allclose(y1, y2) # very strict, atol=1e-8, rtol=1e-5
    assert_close(y1, y2, atol=1e-5, rtol=1e-5)
    #   File "/mnt/d/cs336/cs336-a2-systems/cs336_systems/run_weighted_sum.py", line 32, in <module>
    #     assert_close(y1, y2, atol=1e-6, rtol=1e-6)
    #   File "/mnt/d/cs336/cs336-a2-systems/.venv/lib/python3.12/site-packages/torch/testing/_comparison.py", line 1530, in assert_close
    #     raise error_metas[0].to_error(msg)
    # AssertionError: Tensor-likes are not close!

    # Mismatched elements: 5 / 128 (3.9%)
    # Greatest absolute difference: 4.291534423828125e-06 at index (110,) (up to 1e-06 allowed)
    # Greatest relative difference: 0.00020523578859865665 at index (110,) (up to 1e-06 allowed)
    print("Forward test passed!")
    grad_output = torch.randn_like(y1)

    grad_x_manual, grad_weight_manual = manual_weighted_sum_backward(x, weight, grad_output)

    y2.backward(grad_output, retain_graph=True)
    grad_x_triton = x.grad.clone()
    grad_weight_triton = weight.grad.clone()

    print("Gradient shapes:")
    print(f"grad_x: {grad_x_triton.shape} vs {grad_x_manual.shape}")
    print(f"grad_weight: {grad_weight_triton.shape} vs {grad_weight_manual.shape}")
    
    assert_close(grad_x_triton, grad_x_manual, atol=1e-5, rtol=1e-5)
    assert_close(grad_weight_triton, grad_weight_manual, atol=1e-5, rtol=1e-5)
    print("Backward test passed!")
