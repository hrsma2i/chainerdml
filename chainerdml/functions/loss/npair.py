import chainer.functions as F
from chainer import cuda
from chainer import using_config

from chainerdml.functions.math.euclidean_pairwise_distances import euclidean_pairwise_distances
from chainerdml.functions.loss.npair_hinge import npair_hinge


def npair(
        u, v, labels=None,
        metric_type='dot', loss_type='hinge', l2_reg=3e-3,
        margin=187.0, bidirectional=False, mult_margin=False,
        return_apn_norm=False):

    _valid_metric_types = ['dot', 'cosine', 'euclidean']
    if metric_type not in _valid_metric_types:
        raise ValueError(
            '`metric_type` must be selected from {}.'.format(
                ', '.join(_valid_metric_types)))

    if metric_type == 'cosine':
        u = F.normalize(u, axis=1)
        v = F.normalize(v, axis=1)

    metrics = _calc_metric_matrix(u, v, metric_type, bidirectional)
    loss = _calc_loss(metrics, labels, loss_type, bidirectional)

    if not return_apn_norm:
        return loss
    else:
        ap, an, pnu, pnv =\
            _get_anc_pos_neg(metrics, u, v, metric_type, labels, bidirectional)
        norm_u, norm_v = _get_norms(u, v)
        return loss, ap, an, pnu, pnv, norm_u, norm_v


def _calc_metric_matrix(u, v, metric_type, bidirectional):
    if metric_type == 'euclidean':
        # distance
        if not bidirectional:
            raise NotImplementedError
        metrics = euclidean_pairwise_distances(u, v)
    else:
        # similarity
        metrics = F.matmul(u, v, transb=True)
    # metrics: (B_u, B_v)
    return metrics


def _calc_loss(metrics, labels, loss_type, bidirectional):
    if not bidirectional:
        if labels is None:
            raise ValueError('labels is needed when not bidirectional.')

        if loss_type == 'hinge':
            loss = npair_hinge(metrics, labels)
        else:
            raise ValueError('Unknown loss_type {}.'.format(loss_type))
    else:
        B_u, B_v = metrics.shape
        if B_u == B_v:
            raise ValueError('B_u and B_v must be the same when bidirectional.')

        xp = cuda.get_array_module(metrics)
        labels = xp.arange(B_u)
        # labels: (B_u, )

        if loss_type == 'hinge':
            loss_u = npair_hinge(metrics, labels)
            loss_v = npair_hinge(metrics.T, labels)
            loss = loss_u + loss_v
        else:
            raise ValueError('Unknown loss_type {}.'.format(loss_type))

    return loss


def _get_anc_pos_neg(metrics, u, v, metric_type, labels, bidirectional):
    """
    Returns:
        (dict<xp.array>)
            anc_pos (xp.array (1, ))
            anc_neg (xp.array (1, ))
            pos_neg_u (xp.array (1, ))
            pos_neg_v (xp.array (1, ))
    """
    xp = cuda.get_array_module(metrics)

    with using_config('train', False),\
            using_config('enable_backprop', False):

        B_u, B_v = metrics.shape
        if bidirectional:
            anc_pos = F.mean(F.diagonal(metrics))

            # multiply zero to diagonal elements of similarities (sim_ap)
            mask = xp.ones_like(metrics) - xp.eye(B_u, dtype=metrics.dtype)
        else:
            # arange_B_u: [0, 1, ..., B_u-1]
            # metrics between anchor and positive samples
            metric_ap = metrics[xp.arange(B_u), labels].reshape(-1, 1)
            # metric_ap: (B_u, 1)
            anc_pos = F.mean(metric_ap)

            # for multiplying zero to metric_ap
            mask = xp.ones_like(metrics)
            mask[xp.arange(B_u), labels] = 0
            # mask: (B_u, B_v)

        anc_neg = F.sum(mask * metrics) / (B_u * (B_v - 1))

        metrics_u = _calc_metric_matrix(u, u)
        mask_u = xp.ones_like(metrics_u)
        mask_u[xp.arange(B_u), xp.arange(B_u)] = 0
        pos_neg_u = F.sum(mask_u * metrics_u) / (B_u * (B_u - 1))

        metrics_v = _calc_metric_matrix(v, v)
        mask_v = xp.ones_like(metrics_v)
        mask_v[xp.arange(B_v), xp.arange(B_v)] = 0
        pos_neg_v = F.sum(mask_v * metrics_v) / (B_v * (B_v - 1))

        if metric_type == 'euclidean':
            anc_pos = F.sqrt(anc_pos)
            anc_neg = F.sqrt(anc_neg)
            pos_neg_u = F.sqrt(pos_neg_u)
            pos_neg_v = F.sqrt(pos_neg_v)

        anc_pos = anc_pos.data
        anc_neg = anc_neg.data
        pos_neg_u = pos_neg_u.data
        pos_neg_v = pos_neg_v.data

    return anc_pos, anc_neg, pos_neg_u, pos_neg_v


def _get_norms(u, v):
    """
    Returns:
        norm_u (xp.array (1, ))
        norm_v (xp.array (1, ))
    """
    with using_config('train', False),\
            using_config('enable_backprop', False):

        norm_u = F.mean(F.sqrt(F.batch_l2_norm_squared(u)))
        norm_v = F.mean(F.sqrt(F.batch_l2_norm_squared(v)))

    return norm_u, norm_v

