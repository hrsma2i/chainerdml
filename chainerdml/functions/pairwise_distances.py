import chainer.functions as F


def euclidean_pairwise_distances(u, v):
    """
    This is not needed when applying l2 normalization.
    Args:
        u (chainer.Variable or xp.ndarray:(b, emb))
        v (chainer.Variable or xp.ndarray:(b, emb))
    """
    B = u.shape[0]
    u2 = F.batch_l2_norm_squared(u).reshape(-1, 1)
    # u2: (b, 1)
    u2 = F.broadcast_to(u2, (B, B))
    # u2: (b, b)

    v2 = F.batch_l2_norm_squared(v).reshape(1, -1)
    # v2: (1, b)
    v2 = F.broadcast_to(v2, (B, B))
    # v2: (b, b)

    uv = F.matmul(u, v, transb=True)
    # uv: (b, b)

    distance = u2 - 2.0 * uv + v2
    # distance: (b, b)

    return distance
