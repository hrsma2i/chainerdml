import chainer.functions as F
from chainer import cuda


def _lossfun_hinge_bidir(metrics, metric_type, margin, mult_margin=False):
    """
    Args:
        metrics (chainer.Variable or xp.ndarray:(b, b))
        metric_type: (str) 'euclidean', 'cosine', 'dot', 'dot_l2reg'
        margin (float)
    """
    xp = cuda.get_array_module(metrics)

    # metrics between anchor and positive
    metric_ap = F.diagonal(metrics)
    # metric_ap: (b, )
    metric_ap_hstack = F.broadcast_to(metric_ap, metrics.shape)
    metric_ap_vstack = metric_ap_hstack.transpose(1, 0)
    # metric_ap_*: (b, b)

    # multiply zero to diagonal elements of metrics (metric_ap)
    mask = xp.ones_like(metrics) - xp.eye(metrics.shape[0],
                                          dtype=metrics.dtype)
    # mask: (b, b)

    if metric_type == 'euclidean':
        d_neg = metrics
        d_pos_u = metric_ap_hstack
        d_pos_v = metric_ap_vstack

        if mult_margin:
            cost_u = mask * F.relu(- d_neg + margin*d_pos_u)
            cost_v = mask * F.relu(- d_neg + margin*d_pos_v)
        else:
            cost_u = mask * F.relu(margin - d_neg + d_pos_u)
            cost_v = mask * F.relu(margin - d_neg + d_pos_v)
        # cost_*: (b, b)
        cost_u = F.sqrt(F.mean(cost_u))
        cost_v = F.sqrt(F.mean(cost_v))
    else:
        s_neg = metrics
        s_pos_u = metric_ap_hstack
        s_pos_v = metric_ap_vstack

        if mult_margin:
            min_m = 0.8/margin
            cost_u = mask * F.relu(
                # margin * max(metrics, min_m)
                # if sim_np gets smaller than min_m,
                # sim_ap keeps larger than `min_m*margin`.
                margin*(F.relu(s_neg-min_m)+min_m)
                - s_pos_u)
            cost_v = mask * F.relu(
                # margin * max(metrics, min_m)
                margin*(F.relu(s_neg-min_m)+min_m)
                - s_pos_v)
        else:
            cost_u = mask * F.relu(margin + s_neg - s_pos_u)
            cost_v = mask * F.relu(margin + s_neg - s_pos_v)
        # cost_*: (b, b)
        cost_u = F.mean(cost_u)
        cost_v = F.mean(cost_v)

    loss_hinge = cost_u + cost_v

    return loss_hinge


def npair_hinge(
        metrics, labels, metric_type, margin, mult_margin=False):
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

        if mult_margin:
            raise NotImplementedError
            # if mult_margin:
            #     cost_u = mask * F.relu(- metrics + margin*metric_ap_hstack)
            #     cost_v = mask * F.relu(- metrics + margin*metric_ap_vstack)
            # else:
            #     cost_u = mask * F.relu(margin - metrics + metric_ap_hstack)
            #     cost_v = mask * F.relu(margin - metrics + metric_ap_vstack)
            # # cost_*: (b, b)
            # cost_u = F.sqrt(F.mean(cost_u))
            # cost_v = F.sqrt(F.mean(cost_v))
        else:
            loss = F.mean(mask * F.relu(- d_neg + d_pos + margin))
    else:
        s_neg = metrics
        s_pos = metric_ap_hstack

        if mult_margin:
            raise NotImplementedError
            # min_m = 0.8/margin
            # cost_u = mask * F.relu(
            #     # margin * max(metrics, min_m)
            #     # if sim_np gets smaller than min_m,
            #     # sim_ap keeps larger than `min_m*margin`.
            #     margin*(F.relu(metrics-min_m)+min_m)
            #     - metric_ap_hstack)
            # cost_v = mask * F.relu(
            #     # margin * max(metrics, min_m)
            #     margin*(F.relu(metrics-min_m)+min_m)
            #     - metric_ap_vstack)
        else:
            loss = F.mean(mask * F.relu(s_neg - s_pos + margin))

    return loss
