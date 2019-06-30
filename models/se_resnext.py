import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv.links import Conv2DBNActiv
from chainercv.links.model.resnet import ResBlock
from chainercv.links import PickableSequentialChain


class SEResNeXt(PickableSequentialChain):

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self, n_layer,
                 n_class=None,
                 pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        blocks = self._blocks[n_layer]
        self.mean = mean

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initializers.Normal(scale=0.01)

        kwargs = {
            'groups': 32, 'initialW': initialW, 'stride_first': False,
            'add_seblock': True}

        super(SEResNeXt, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 3, 1, 1, nobias=True,
                                       initialW=initialW)
            self.res2 = ResBlock(blocks[0], None, 128, 256, 2, **kwargs)
            self.res3 = ResBlock(blocks[1], None, 256, 512, 1, **kwargs)
            self.res4 = ResBlock(blocks[2], None, 512, 1024, 2, **kwargs)
            self.res5 = ResBlock(blocks[3], None, 1024, 2048, 1, **kwargs)
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(None, n_class, **fc_kwargs)


class SEResNeXt50(SEResNeXt):

    def __init__(self, n_class=10, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNeXt50, self).__init__(
            50, n_class, pretrained_model,
            mean, initialW, fc_kwargs)


class SEResNeXt101(SEResNeXt):

    def __init__(self, n_class=10, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNeXt101, self).__init__(
            101, n_class, pretrained_model,
            mean, initialW, fc_kwargs)


class SEResNeXt152(SEResNeXt):

    def __init__(self, n_class=10, pretrained_model=None,
                 mean=None, initialW=None, fc_kwargs={}):
        super(SEResNeXt152, self).__init__(
            152, n_class, pretrained_model,
            mean, initialW, fc_kwargs)
