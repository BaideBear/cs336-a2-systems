import torch
import triton
import triton.language as tl

@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets<num_cols, other=float('-inf'))

    x_row = x_row - tl.max(x_row, axis=0)
    exp_x_row = tl.exp(x_row)
    exp_sum = tl.sum(exp_x_row, axis=0)
    y_row = exp_x_row / exp_sum

    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets<num_cols)



def triton_softmax(x: torch.Tensor):
    y = torch.empty_like(x)

    M, N = x.shape
    # triton.next_power_of_2(N) 的作用是计算大于等于N的最小2的幂次方数
    block_size = triton.next_power_of_2(N) #每个block处理一整行
    num_blocks = M 

    triton_softmax_kernel[(M,)](x_ptr=x, y_ptr=y, x_row_stride=x.stride(0),
                                y_row_stride=y.stride(0),
                                num_cols=N, BLOCK_SIZE=block_size)
    
    return y
