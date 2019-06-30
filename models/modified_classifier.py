from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link, cross_entropy, backends
from chainer import reporter
import chainer.functions as F


class ModifiedClassifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(ModifiedClassifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def forward(self, *args, **kwargs):

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.predictor(*args, **kwargs)

        xp = backends.cuda.get_array_module(t)
        t1 = t[:, 0].astype(xp.int32)
        t2 = t[:, 1].astype(xp.int32)
        ratio1 = t[:, 2].astype(xp.float32)
        ratio2 = t[:, 3].astype(xp.float32)

        self.loss = self.lossfun(self.y, t1, reduce='no') * ratio1 + \
            self.lossfun(self.y, t2, reduce='no') * ratio2
        self.loss = F.mean(self.loss)
        reporter.report({'loss': self.loss}, self)
        gt = t1 * (ratio1 >= ratio2) + t2 * (ratio1 < ratio2)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, gt)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
