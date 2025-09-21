import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
import os

class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x
    
def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())

    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()

            # Backward
            y.backward()

    return run


def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x, y)

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Print out table
    table = prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total",
        max_name_column_width=80,
        row_limit=10
    )

    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    print(table)

    return table


def profile_cuda(description: str, run: Callable):
    # Warmup
    for _ in range(1):
        run()
    torch.cuda.synchronize()

    # Measure CUDA time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    run()
    end_event.record()
    torch.cuda.synchronize()  # Wait for CUDA threads to finish

    # Print CUDA time
    cuda_time = start_event.elapsed_time(end_event)
    print(f"{description} CUDA time: {cuda_time:.2f} ms")

def profiling():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    # sleep_function = lambda : time.sleep(50 / 1000)
    # sleep_profile = profile("sleep", sleep_function) 

    # add_function = lambda a, b: a + b
    # add_profile = profile("add", run_operation2(dim=2048, operation=add_function))

    # matmul_function = lambda a, b: a @ b
    # matmul_profile = profile("matmul", run_operation2(dim=8192, operation=matmul_function))
    # profile_cuda("matmul", run_operation2(dim=8192, operation=matmul_function))

    # gelu_function = lambda a, b: torch.nn.functional.gelu(a + b)
    # gelu_profile = profile("gelu", run_operation2(dim=2048, operation=gelu_function))

    if torch.cuda.is_available():
        mlp_profile = profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2))
    else:
        mlp_profile = profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2))



if __name__ == "__main__":
    profiling()