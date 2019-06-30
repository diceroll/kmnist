import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from models.connections.conv_2d_bn_activ import Conv2DBNActiv
from models.connections.resblock import ResBlock
from chainercv.links import PickableSequentialChain


class SERes2Net(PickableSequentialChain):

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self, n_layer,
                 n_class=None, scale=4,
                 pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        blocks = self._blocks[n_layer]
        self.mean = mean

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initializers.Normal(scale=0.01)

        kwargs = {
            'scale': scale, 'initialW': initialW, 'stride_first': True,
            'add_seblock': True}

        super(SERes2Net, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 3, 1, 1, nobias=True,
                                       initialW=initialW)
            self.res2 = ResBlock(blocks[0], None, 64, 256, 2, **kwargs)
            self.res3 = ResBlock(blocks[1], None, 128, 512, 1, **kwargs)
            self.res4 = ResBlock(blocks[2], None, 256, 1024, 2, **kwargs)
            self.res5 = ResBlock(blocks[3], None, 512, 2048, 1, **kwargs)
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(None, n_class, **fc_kwargs)


class SERes2Net50(SERes2Net):

    def __init__(self, n_class=10, scale=4, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SERes2Net50, self).__init__(
            50, n_class, scale, pretrained_model,
            mean, initialW, fc_kwargs)


class SERes2Net101(SERes2Net):

    def __init__(self, n_class=10, scale=4, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SERes2Net101, self).__init__(
            101, n_class, scale, pretrained_model,
            mean, initialW, fc_kwargs)


class SERes2Net152(SERes2Net):

    def __init__(self, n_class=10, scale=4, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SERes2Net152, self).__init__(
            152, n_class, scale, pretrained_model,
            mean, initialW, fc_kwargs)
