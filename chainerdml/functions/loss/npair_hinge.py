import chainer.functions as F
from chainer import cuda


def npair_hinge(metrics, labels, metric_type, margin):
    """
    Args:
        metrics (chainer.Variable or xp.ndarray:(B, N))
            B: A mini batch size.
            N: The number of positive and negative samples.
        labels (xp.ndarray<int>: (B, )):
            Positive pair column indices.
            Each element ranges in [0, N-1].
        metric_type: (str) 'euclidean', 'cosine', 'dot', 'dot_l2reg'
        margin (float)
    """
    xp = cuda.get_array_module(metrics)

    arange_B = xp.arange(metrics.shape[0])
    # arange_B: [0, 1, ..., B-1]
    # metrics between anchor and positive samples
    metric_ap = metrics[arange_B, labels].reshape(-1, 1)
    # metric_ap: (B, 1)
    metric_ap_hstack = F.broadcast_to(metric_ap, metrics.shape)
    # metric_ap_*: (B, N)

    # for multiplying zero to metric_ap
    mask = xp.ones_like(metrics)
    mask[arange_B, labels] = 0
    # mask: (B, N)

    if metric_type == 'euclidean':
        d_neg = metrics
        d_pos = metric_ap_hstack

        loss = F.mean(mask * F.relu(- d_neg + d_pos + margin))
    elif metric_type == 'euclidean':
        s_neg = metrics
        s_pos = metric_ap_hstack

        loss = F.mean(mask * F.relu(s_neg - s_pos + margin))
    else:
        ValueError('Unknown metric_type, {}'.format(metric_type))

    return loss
