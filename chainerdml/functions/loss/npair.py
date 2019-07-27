import chainer.functions as F
from chainer import cuda
from chainer import using_config

from chainerdml.functions import euclidean_pairwise_distances
from chainerdml.functions import npair_hinge


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


class NpairLoss:
    _valid_metric_types = ['dot', 'cosine', 'euclidean']

    def __init__(
        self,
        model,
        metric_type='dot',
        # euclidean
        # dot (inner product)
        # dot_l2reg (l2 regularized)
        l2_reg=3e-3,
        loss_type='hinge',
        margin=187.0,
        bidirectional=False,
        mult_margin=False,
        is_first_iter=True,
        init_loss_ratio=None,
        cache=False,
    ):
        """

        Args:
            model (chainer.Chain): A model to which is added this loss.
            metric_type (str, optional): Defaults to 'dot'.
                `dot`, `cosine`, or `euclidean`.
            loss_type (str, optional): [description]. Defaults to 'hinge'.
            margin (float, optional): [description]. Defaults to 187.0.
            bidirectional (bool, optional): [description]. Defaults to False.
                If this is `True`, use both losses
                e.g., use the loss whose anchor is from the domain U,
                and the loss whose anchor is from the domain V.
            mult_margin (bool, optional): [description]. Defaults to False.
            is_first_iter (bool, optional): [description]. Defaults to True.
            init_loss_ratio ([type], optional): [description]. Defaults to None.
            cache (bool, optional): [description]. Defaults to False.
        """
        self.model = model
        if metric_type not in self._valid_metric_types:
            raise ValueError(
                '`metric_type` must be selected from {}.'.format(
                    ', '.join(self._valid_metric_types)
                )
            )
        self.metric_type = metric_type
        self.l2_normalize = metric_type == 'cosine'
        self.l2_reg = l2_reg
        self.loss_type = loss_type
        self.margin = margin
        self.bidirectional = bidirectional
        self.mult_margin = mult_margin
        self.cache = cache

    def __call__(self, u, v, labels=None):
        """
        Args:
            u (xp.array (b_u, emb)):
                a batch of features
            v (xp.array (b_v, emb)):
                a batch of features
            labels (xp.ndarray: (b_u)):
                Positive pair column indices.
                Each value is ranged in [0, b_v-1].
                This is needed when `bidirectional is False`.
        Return:
            loss (Variable (1, ))
        """

        if self.l2_normalize:
            u = F.normalize(u, axis=1)
            v = F.normalize(v, axis=1)

        metrics = self.calc_metric_matrix(u, v)
        loss = self.lossfun(metrics, labels=labels)

        # for report
        if self.cache:
            self.metrics = metrics
            self.u = u
            self.v = v

        if 'reg' in self.metric_type:
            loss_l2 = lossfun_l2(u, v)
            r = self.get_coef_l2_loss(loss, loss_l2)
            # reporter.report({'loss_l2': loss_l2}, self)
            # reporter.report({'loss_ratio': (r*loss_l2)/loss}, self)

            # loss += l2_reg * loss_l2
            loss += r * loss_l2

        return loss

    def calc_metric_matrix(self, u, v):
        """
        Args:
            u (xp.array (b, emb)):
                a batch of features
            v (xp.array (b, emb)):
                a batch of features
        Return:
            metrics(Variable (b, b)): metric matrix
        """
        if self.metric_type == 'euclidean':
            # distance
            metrics = euclidean_pairwise_distances(u, v)
        else:
            # similarity
            metrics = F.matmul(u, v, transb=True)
        # metrics: (b, b)

        return metrics

    def lossfun(self, metrics, labels=None):
        """
        Args:
            metrics (chainer.Variable: (b_u, b_v))
            labels (xp.ndarray: (b_u)):
                Positive pair column indices.
                Each value is ranged in [0, b_v-1].
                This is needed when `bidirectional is False`.
        Return:
            loss (Variable (1, ))
        """
        metric_type = self.metric_type
        loss_type = self.loss_type
        margin = self.margin
        mult_margin = self.mult_margin

        if self.bidirectional:
            if loss_type == 'sce':
                loss = lossfun_sce_bidir(metrics)
            elif loss_type == 'hinge':
                loss = lossfun_hinge_bidir(
                    metrics, metric_type=metric_type,
                    margin=margin, mult_margin=mult_margin)
        else:
            if loss_type == 'sce':
                loss = F.softmax_cross_entropy(metrics, labels)
            elif loss_type == 'hinge':
                loss = lossfun_hinge(
                    metrics, labels, metric_type=metric_type,
                    margin=margin, mult_margin=mult_margin)
        return loss

    def get_coef_l2_loss(self, loss, loss_l2):
        """
        Args:
            loss (Variable (1, ))
            loss_l2 (Variable (1, ))

        Return:
            (float): the coefficient of L2 regularization loss term
        """
        l2_reg = self.l2_reg

        if self.model.is_first_iter:
            # float is needed to avoid mixing np and cp error
            self.model.init_loss_ratio = float(l2_reg * loss.data/loss_l2.data)
            self.model.is_first_iter = False
        return float(self.model.init_loss_ratio)

    def get_anc_pos_neg(self, labels=None):
        """
        Returns:
            (dict<xp.array>)
                anc_pos (xp.array (1, ))
                anc_neg (xp.array (1, ))
                pos_neg_u (xp.array (1, ))
                pos_neg_v (xp.array (1, ))
        """
        metrics = self.metrics
        # metrics: (b_u, b_v)
        u = self.u
        v = self.v
        # u|v: (b_, emb)

        xp = cuda.get_array_module(metrics)

        with using_config('train', False),\
                using_config('enable_backprop', False):

            B_u, B_v = metrics.shape
            if self.bidirectional:
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

            metrics_u = self.calc_metric_matrix(u, u)
            mask_u = xp.ones_like(metrics_u)
            mask_u[xp.arange(B_u), xp.arange(B_u)] = 0
            pos_neg_u = F.sum(mask_u * metrics_u) / (B_u * (B_u - 1))

            metrics_v = self.calc_metric_matrix(v, v)
            mask_v = xp.ones_like(metrics_v)
            mask_v[xp.arange(B_v), xp.arange(B_v)] = 0
            pos_neg_v = F.sum(mask_v * metrics_v) / (B_v * (B_v - 1))

            if self.metric_type == 'euclidean':
                anc_pos = F.sqrt(anc_pos)
                anc_neg = F.sqrt(anc_neg)
                pos_neg_u = F.sqrt(pos_neg_u)
                pos_neg_v = F.sqrt(pos_neg_v)

            anc_pos = anc_pos.data
            anc_neg = anc_neg.data
            pos_neg_u = pos_neg_u.data
            pos_neg_v = pos_neg_v.data

        return dict(anc_pos=anc_pos,
                    anc_neg=anc_neg,
                    pos_neg_u=pos_neg_u,
                    pos_neg_v=pos_neg_v)

    def get_norms(self):
        """
        Returns:
            norm_u (xp.array (1, ))
            norm_v (xp.array (1, ))
        """
        u = self.u
        v = self.v
        # x_: (b, emb)
        with using_config('train', False),\
                using_config('enable_backprop', False):

            norm_u = F.mean(F.sqrt(F.batch_l2_norm_squared(u)))
            norm_v = F.mean(F.sqrt(F.batch_l2_norm_squared(v)))

        return dict(norm_u=norm_u.data,
                    norm_v=norm_v.data)

    def delete_cache(self):
        # for memory efficency
        del self.u
        del self.v
        del self.metrics


def lossfun_sce_bidir(metrics):
    """
    Softmax Cross Entropy
    `metrics` must be **similarity** matrix.

    Args:
        metrics (chainer.Variable or xp.ndarray:(b, b))
        metric_type: (str) 'euclidean', 'cosine', 'dot', 'dot_l2reg'
        margin (float)
    """
    xp = cuda.get_array_module(metrics)

    B = len(metrics.data)
    cost_u = F.softmax_cross_entropy(metrics, xp.arange(B))
    cost_v = F.softmax_cross_entropy(metrics.T, xp.arange(B))
    loss_sce = (cost_u + cost_v) / 2
    return loss_sce


def lossfun_l2(u, v):
    """
    This makes the norm of u/shop close to 1.
    Use this instead of l2 normalization.
    If you use this, you must share network.

    Args:
        u (chainer.Variable or xp.ndarray:(b, emb))
        v (chainer.Variable or xp.ndarray:(b, emb))
    """
    B = len(u)
    # use relu and marign 1
    # to avoid for norms being less than 1
    loss_l2 = F.relu(-1 +
                     sum(F.batch_l2_norm_squared(u) +
                         F.batch_l2_norm_squared(v)) / (2.0 * B))
    return loss_l2