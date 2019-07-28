# ChainerDML: a Chainer-based library for Deep Metric Learning

```py
from chainerdml.npair_loss import NpairLoss


class NN(chainer.Chain):
    def __init__(self):
        self.lossfun = NpairLoss(
            metric_type='dot',
            loss_type='hinge',
            margin=187.0,
            bidirectional=False,
            cache=False,
        )
    
    def __call__(self, in_data):

        ...

        loss = self.lossfun(u, v, labels)

        reporter.report(self.lossfun.anc_pos_neg)
        reporter.report(self.lossfun.norms)
        self.lossfun.clear_cache()

        return loss
```