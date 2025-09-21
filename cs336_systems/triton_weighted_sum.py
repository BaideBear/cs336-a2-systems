import triton
import triton.language as tl
import torch
from einops import rearrange
from triton import cdiv

"""
    安全地从内存加载数据块，支持边界检查和自动填充
    
    该函数用于从指定的内存位置安全地加载数据块，当访问越界时会自动返回
    填充值（零或NaN）而不是引发内存错误，确保GPU内核的稳定运行。
    
    参数:
    -----------
    ptr : BlockPointer
        要加载的内存块指针，通常通过tl.make_block_ptr()创建。
        包含基地址、形状、步长和偏移量信息。
        x = tl.make_block_ptr(
            base,                    # void*，数据首地址（通常 tensor.data_ptr()+偏移）
            shape,                   # tuple[int, ...]  整个张量的逻辑尺寸
            strides,                 # tuple[int, ...]  每个维度跨多少元素
            offsets,                 # tuple[int, ...]  我要的 tile 起始逻辑坐标
            block_shape,             # tuple[int, ...]  我要的 tile 尺寸
            order                    # tuple[int, ...]  维度在内存中的主序（row-major 为 (1,0)）, 实际含义是告诉我们哪一维度是连续的
        )
        x = x.advance((a, b)) # 在不同维度上stride相应的值
        
    boundary_check : tuple of ints, 可选, 默认=(0, 1)
        需要进行边界检查的维度。
        例如：(0, 1) 表示同时检查第0维和第1维的越界访问
        使用 () 可禁用边界检查（不推荐，影响安全性）
        
    padding_option : str, 可选, 默认="zero"
        越界访问时的填充方式：
        - "zero": 用零填充越界元素
        - "nan": 用NaN填充越界元素
        
    返回值:
    --------
    Tensor
        加载的数据块，形状与ptr中指定的相同
        越界元素会根据padding_option参数进行填充
        
    示例:
    --------
    >>> # 为2D张量创建块指针
    >>> block_ptr = tl.make_block_ptr(
    ...     base=ptr,
    ...     shape=(N, M),
    ...     strides=(M, 1),
    ...     offsets=(start_i, start_j),
    ...     block_shape=(BLOCK_SIZE, BLOCK_SIZE),
    ...     order=(1, 0)
    ... )
    >>> 
    >>> # 安全加载数据（带边界检查）
    >>> data = tl.load(block_ptr, 
    ...                boundary_check=(0, 1), 
    ...                padding_option="zero")
    
    注意:
    ------
    - 在处理变长输入时特别重要
    - 防止处理边缘情况时的内存访问违规
    - 相比不安全的内存访问，只增加极小的开销
    - 特别适用于深度学习算子（卷积、矩阵乘法等）
    - 推荐在生产代码中使用以确保健壮性
    """

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim, #x的一维stride，二维stride
    weight_stride_dim,
    output_stride_dim,
    ROWS, D, # x的形状
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, #两个方向上的tiling_size, 实际上D维度没tiling，只是按2的指数大小填充
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D, ),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx*ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D, ),
        strides=(weight_stride_dim, ),
        offsets=(0, ),
        block_shape=(D_TILE_SIZE, ),
        order=(0, ),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS, ),
        strides=(output_stride_dim, ),
        offsets=(row_tile_idx*ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0, ),
    )

    output = tl.zeros((ROWS_TILE_SIZE, ), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)): 
        #(ROWS_TILE_SIZE, D_TILE_SIZE)
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        #(D_TILE_SIZE, )
        weight = tl.load(weight_block_ptr, boundary_check=(0, ), padding_option="zero")
        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))

    tl.store(output_block_ptr, output, boundary_check=(0, ))

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,
    grad_output_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    stride_x_r, stride_x_d,
    stride_w_d,
    stride_g_r,
    stride_gx_r, stride_gx_d,
    stride_gw_b, stride_gw_d,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS, ), strides=(stride_g_r, ),
        offsets=(row_tile_idx*ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0, ),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D, ), strides=(stride_x_r, stride_x_d), 
        offsets=(row_tile_idx*ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D, ), strides=(stride_w_d, ), 
        offsets=(0, ), block_shape=(D_TILE_SIZE, ),
        order=(0, ),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D, ), strides=(stride_gx_r, stride_gx_d, ),
        offsets=(row_tile_idx*ROWS_TILE_SIZE, 0), 
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE, ),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D, ), strides=(stride_gw_b, stride_gw_d),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # (ROWS_TILE_SIZE,)
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

        # 通过外积求 grad_x
        # (D_TILE_SIZE,)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # (ROWS_TILE_SIZE, D_TILE_SIZE)
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        # 0维度是Thread Block数量，永远不会out of bounds
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1, ))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]

        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        # 保存x, weight, 以便backward使用
        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D
        assert x.is_cuda and weight.is_cuda
        assert x.is_contiguous()

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        #后面记得试一下这里把数据展平会发生什么
        y = torch.empty(output_dims, device=x.device)

        n_rows = y.numel()
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE), )](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        #这里也要测试，kernel是否改变了torch张量储存额形状
        return y.view(input_shape[:-1])

    @staticmethod     
    def backward(ctx, grad_outputs):         
        x, weight = ctx.saved_tensors         
        # 没必要是一样的         
        D_TILE_SIZE = ctx.D_TILE_SIZE
        ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE
        n_rows, D = x.shape
        # 策略是先都存放到buffer中，最后做reduce
        partial_grad_weight = torch.empty((cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        #kernel
        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE), )](
            x, weight,
            grad_outputs,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_outputs.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight

        
