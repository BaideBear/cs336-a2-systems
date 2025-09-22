import triton
import triton.language as tl
import torch

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, 
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scales,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx*stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq, ),
        offsets=(query_tile_idx*Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )

    # Q_i: [Q_TILE_SIZE, D]
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, ), padding_option='zero')
    # l_i: [Q_TILE_SIZE, ]
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    # m_i: [Q_TILE_SIZE, ]
    m_i = tl.full((Q_TILE_SIZE, ), float('-inf'), dtype=tl.float32)
    # O_i: [Q_TILE_SIZE, D]
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    log2e: tl.constexpr = 1.44269504
    eps: tl.constexpr = 1e-6

    if is_causal:
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_idx*Q_TILE_SIZE

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # K_j: [K_TILE_SIZE, D]
        K_j = tl.load(K_block_ptr, boundary_check=(0, ), padding_option="zero")
        # V_j: [K_TILE_SIZE, D]
        V_j = tl.load(V_block_ptr, boundary_check=(0, ), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scales 

        # causal mask
        if is_causal:
            k_pos = tl.arange(0, K_TILE_SIZE) + j*K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :]
            S_ij = tl.where(mask, S_ij, float('-inf'))

        m_cur = tl.maximum(m_i, tl.max(S_ij, axis=-1))
        # exp2具有更好的数值稳定性
        P_ij = tl.math.exp2((S_ij - m_cur[:, None]) * log2e)

        alpha = tl.math.exp2((m_i - m_cur) * log2e)
        m_i = m_cur
        l_i = alpha * l_i + tl.sum(P_ij, axis=-1)
        O_i = O_i * alpha[:, None] + tl.dot(P_ij.to(V_j.dtype), V_j)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    O_i = O_i / (l_i[:, None] + eps)
    l_i = m_i + tl.math.log2(l_i)/log2e

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, ))
    tl.store(L_block_ptr, l_i.to(L_block_ptr.type.element_ty), boundary_check=(0, ))

@triton.jit
def flash_bwd_cal_D_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERYS, D_,
    Q_TILE_SIZE, D_TILE_SIZE,
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx*stride_ob,
        shape=(N_QUERYS, D_),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx*stride_dob,
        shape=(N_QUERYS, D_),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_idx*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx*stride_db,
        shape=(N_QUERYS, ),
        strides=(stride_dq, ),
        offsets=(query_tile_idx*Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )

    D = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)

    for i in range(tl.cdiv(D_, D_TILE_SIZE)):
        o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        D += tl.sum(o * do, axis=1)
        O_block_ptr = O_block_ptr.advance((0, D_TILE_SIZE))
        dO_block_ptr = dO_block_ptr.advance((0, D_TILE_SIZE))

    tl.store(D_block_ptr, D, boundary_check=(0, ))

@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, 
    dO_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS, 
    scales,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx*stride_db,
        shape=(N_QUERIES, ),
        strides=(stride_dq, ),
        offsets=(query_tile_idx*Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx*stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_idx*Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx*stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_idx*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_idx*stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_idx*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, ), padding_option="zero").to(tl.float32) 
    L_i = tl.load(L_block_ptr, boundary_check=(0, ), padding_option="zero")
    D_i = tl.load(D_block_ptr, boundary_check=(0, ), padding_option="zero")
    dO_i = tl.load(dO_block_ptr, boundary_check=(0, ), padding_option="zero").to(tl.float32)
    dQ_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    log2e: tl.constexpr = 1.44269504
    eps: tl.constexpr = 1e-6

    if is_causal:
        q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_idx*Q_TILE_SIZE

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr, boundary_check=(0, ), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, ), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scales
        if is_causal:
            k_pos = tl.arange(0, K_TILE_SIZE) + j*K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :]
            S_ij = tl.where(mask, S_ij, float('-inf'))
        P_ij = tl.exp2((S_ij - L_i[:, None]) * log2e)

        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = P_ij * (dP_ij - D_i[:, None])
        dQ_i += tl.dot(dS_ij, K_j) * scales

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    tl.store(dQ_block_ptr, dQ_i.to(dQ_block_ptr.type.element_ty), boundary_check=(0, ))

@triton.jit
def flash_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr,
    D_ptr, L_ptr, 
    dO_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scales,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    key_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_idx*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_idx*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx*stride_db,
        shape=(N_QUERIES, ),
        strides=(stride_dq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx*stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx*stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_idx*stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_idx*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_idx*stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_idx*K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    K_j = tl.load(K_block_ptr, boundary_check=(0, ), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check=(0, ), padding_option="zero")
    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    log2e: tl.constexpr = 1.44269504
    eps: tl.constexpr = 1e-6

    if is_causal:
        k_pos = tl.arange(0, K_TILE_SIZE) + key_tile_idx*K_TILE_SIZE
    
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check=(0, ), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0, ), padding_option="zero")
        D_i = tl.load(D_block_ptr, boundary_check=(0, ), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check=(0, ), padding_option="zero")

        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scales
        if is_causal:
            q_pos = tl.arange(0, Q_TILE_SIZE) + i*Q_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :]
            S_ij = tl.where(mask, S_ij, float('-inf'))
        P_ij = tl.exp2((S_ij - L_i[:, None]) * log2e)

        dV_j += tl.dot(tl.trans(P_ij).to(dO_block_ptr.type.element_ty), dO_i)
        dP_ij = tl.dot(dO_i, tl.trans(V_j))
        dS_ij = (P_ij * (dP_ij - D_i[:, None]))
        dK_j += tl.dot(tl.trans(dS_ij.to(Q_block_ptr.type.element_ty)), Q_i) * scales

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE, ))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE, ))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
    
    tl.store(dK_block_ptr, dK_j.to(dK_block_ptr.type.element_ty), boundary_check=(0, ))
    tl.store(dV_block_ptr, dV_j.to(dV_block_ptr.type.element_ty), boundary_check=(0, ))



class FlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size , d_q, d = Q.shape
        d_k = K.size(1)
        assert d_q == d_k
        # 每个tiling分块的大小
        B_q, B_k = 16, 16
        # 分块的总数
        T_q = triton.cdiv(d_q, B_q)
        T_k = triton.cdiv(d_k, B_k)

        scales = 1 / (d ** 0.5)

        O = torch.empty_like(Q)
        L = torch.empty((batch_size, d_q), device=Q.device)

        flash_fwd_kernel[(T_q, batch_size, )](
            Q, K, V,
            O, L, 
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2), 
            O.stride(0), O.stride(1), O.stride(2), 
            L.stride(0), L.stride(1),
            N_QUERIES=d_q, N_KEYS=d_k,
            scales=scales,
            D=d, Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k,
            is_causal=is_causal,
        )

        #为backwards保存上下文
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.B_q = B_q
        ctx.B_k = B_k
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        B_q = ctx.B_q
        B_k = ctx.B_k
        is_causal = ctx.is_causal
        Q, K, V, O, L = ctx.saved_tensors
        batch_size, d_q, d = Q.shape
        d_k = K.size(1)
        scales = 1 / (d ** 0.5)
        T_q = triton.cdiv(d_q, B_q)
        T_k = triton.cdiv(d_k, B_k)
        assert B_q == B_k

        #计算D = rowsum(dO * O)
        D = torch.sum(O*dO, dim = -1)

        # D = torch.empty_like(L)
        # Q_TILE_SIZE = 64
        # D_TILE_SIZE = 64
        # flash_bwd_cal_D_kernel[(triton.cdiv(d_q, Q_TILE_SIZE), batch_size)](
        #     O, dO, D,
        #     O.stride(0), O.stride(1), O.stride(2),
        #     dO.stride(0), dO.stride(1), dO.stride(2),
        #     D.stride(0), D.stride(1),
        #     N_QUERYS=d_q, D_=d,
        #     Q_TILE_SIZE=Q_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        # )

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        flash_bwd_dq_kernel[(T_q, batch_size)](
            Q, K, V,
            D, L, 
            dO, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_QUERIES=d_q, N_KEYS=d_k,
            scales=scales, D=d,
            Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k,
            is_causal=is_causal
        )

        flash_bwd_dkv_kernel[(T_k, batch_size)](
            Q, K, V,
            D, L, 
            dO, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            D.stride(0), D.stride(1),
            L.stride(0), L.stride(1),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES=d_q, N_KEYS=d_k,
            scales=scales, D=d,
            Q_TILE_SIZE=B_q, K_TILE_SIZE=B_k,
            is_causal=is_causal,
        )

        return dQ, dK, dV, None





