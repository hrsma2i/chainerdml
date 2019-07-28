import chainer.functions as F
from chainer import cuda
from chainer import using_config
from chainerdml.functions import euclidean_pairwise_distances
from chainerdml.functions import npair_hinge


class NpairLoss:
    _valid_metric_types = ['dot', 'cosine', 'euclidean']
    _valid_loss_types = ['hinge', 'softmax']

    def __init__(
        self,
        model,
        metric_type='dot',
        loss_type='hinge',
        margin=187.0,
        bidirectional=False,
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
            cache (bool, optional): [description]. Defaults to False.
        """
        self.model = model
        self.metric_type = metric_type
        self.l2_normalize = metric_type == 'cosine'
        self.loss_type = loss_type
        self.margin = margin
        self.bidirectional = bidirectional
        self.cache = cache

        self._verify()

    def _verify(self):
        checklist = [
            (
                self.metric_type not in self._valid_metric_types,
                ValueError(
                    '`metric_type` must be selected from {}.'.format(
                        ', '.join(self._valid_metric_types)))
            ),
            (
                self.loss_type not in self._valid_loss_types,
                ValueError(
                    '`loss_type` must be selected from {}.'.format(
                        ', '.join(self._valid_loss_types)))
            ),
        ]

        for cond, err in checklist:
            if cond:
                raise err

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
            self.lebels = labels
            self.u = u
            self.v = v

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
            if not self.bidirectional:
                raise NotImplementedError

            metrics = euclidean_pairwise_distances(u, v)
        else:
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

        if self.bidirectional:
            B_u, B_v = metrics.shape
            if B_u == B_v:
                raise ValueError(
                    'B_u and B_v must be the same when bidirectional.')
            xp = cuda.get_array_module(metrics)
            labels = xp.arange(B_u)
            # labels: (B_u, )

        # anchor is u.
        if loss_type == 'hinge':
            loss = npair_hinge(
                metrics, labels, metric_type, margin)
        elif loss_type == 'softmax':
            if metric_type not in ['dot', 'cosine']:
                raise ValueError(
                    'metric_type must be a kind of similarties\
                    when loss_type is `softmax`')
            loss = F.softmax_cross_entropy(metrics, labels)

        # anchor is v.
        if self.bidirectional:
            if loss_type == 'hinge':
                loss += npair_hinge(
                    metrics.T, labels, metric_type, margin)
            elif loss_type == 'softmax':
                loss += F.softmax_cross_entropy(metrics.T, labels)
            loss /= 2

        return loss

    @property
    def anc_pos_neg(self):
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
        labels = self.labels

        xp = cuda.get_array_module(metrics)

        with using_config('train', False),\
                using_config('enable_backprop', False):

            B_u, B_v = metrics.shape
            if self.bidirectional:
                anc_pos = F.mean(F.diagonal(metrics))

                # multiply zero to diagonal elements of similarities (sim_ap)
                mask = xp.ones_like(metrics) - xp.eye(B_u, dtype=metrics.dtype)
            else:
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

        return anc_pos, anc_neg, pos_neg_u, pos_neg_v

    @property
    def norms(self):
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

            norm_u = F.mean(F.sqrt(F.batch_l2_norm_squared(u))).data
            norm_v = F.mean(F.sqrt(F.batch_l2_norm_squared(v))).data

        return norm_u, norm_v

    def clear_cache(self):
        # for memory efficency
        del self.u
        del self.v
        del self.metrics
        del self.labels
