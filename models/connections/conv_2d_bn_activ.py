import chainer
from chainer.functions import relu
from chainer.links import BatchNormalization, Convolution2D

from models.connections.res2net_conv import Res2NetConv
from models.connections.attention_augmented_conv import AugmentedConv

try:
    from chainermn.links import MultiNodeBatchNormalization
except ImportError:
    pass


class Conv2DBNActiv(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize=None,
                 stride=1, pad=0, scale=1, dilate=1, groups=1, nobias=True,
                 initialW=None, initial_bias=None, activ=relu,
                 bn_kwargs={}, aa_kwargs={}):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.activ = activ
        super(Conv2DBNActiv, self).__init__()
        with self.init_scope():
            if len(aa_kwargs) > 0:
                self.conv = AugmentedConv(
                    in_channels, out_channels, ksize,
                    int(out_channels * aa_kwargs['k']),
                    int(out_channels * aa_kwargs['v']),
                    aa_kwargs['Nh'], aa_kwargs['relative'],
                    initialW=initialW)
            elif scale > 1:
                self.conv = Res2NetConv(
                    in_channels, out_channels, ksize, stride, pad, scale,
                    nobias, initialW, initial_bias,
                    dilate=dilate, groups=groups)
            else:
                self.conv = Convolution2D(
                    in_channels, out_channels, ksize, stride, pad,
                    nobias, initialW, initial_bias,
                    dilate=dilate, groups=groups)
            if 'comm' in bn_kwargs:
                self.bn = MultiNodeBatchNormalization(
                    out_channels, **bn_kwargs)
            else:
                self.bn = BatchNormalization(out_channels, **bn_kwargs)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activ is None:
            return h
        else:
            return self.activ(h)
