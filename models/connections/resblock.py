import chainer
import chainer.functions as F

from models.connections.conv_2d_bn_activ import Conv2DBNActiv
from models.connections.gcblock import GCBlock
from chainercv.links import PickableSequentialChain
from chainercv.links import SEBlock


class ResBlock(PickableSequentialChain):

    def __init__(self, n_layer, in_channels, mid_channels, out_channels,
                 stride, scale=1, dilate=1, groups=1,
                 initialW=None, bn_kwargs={}, stride_first=False,
                 add_block=None, aa_kwargs={}):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.a = Bottleneck(
                in_channels, mid_channels, out_channels,
                stride, scale, dilate, groups, initialW, bn_kwargs=bn_kwargs,
                residual_conv=True, stride_first=stride_first,
                add_block=add_block, aa_kwargs={})
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = Bottleneck(
                    out_channels, mid_channels, out_channels,
                    stride=1, scale=scale, dilate=dilate,
                    groups=groups, initialW=initialW,
                    bn_kwargs=bn_kwargs, residual_conv=False,
                    add_block=add_block, aa_kwargs={})
                self.add_link(name, bottleneck)


class Bottleneck(chainer.Chain):

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=1, scale=1, dilate=1, groups=1,
                 initialW=None, bn_kwargs={}, residual_conv=False,
                 stride_first=False, add_block=None, aa_kwargs={}):

        if stride_first:
            first_stride = stride
            second_stride = 1
        else:
            first_stride = 1
            second_stride = stride
        super(Bottleneck, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(in_channels, mid_channels,
                                       1, first_stride, 0,
                                       nobias=True, initialW=initialW,
                                       bn_kwargs=bn_kwargs)
            if len(aa_kwargs) > 0:
                self.conv2 = Conv2DBNActiv(mid_channels, mid_channels,
                                           3, 1, 1, 1, 1, 1, nobias=True,
                                           initialW=initialW,
                                           bn_kwargs=bn_kwargs,
                                           aa_kwargs=aa_kwargs)
            elif stride > 1:
                self.conv2 = Conv2DBNActiv(mid_channels, mid_channels,
                                           3, second_stride, dilate,
                                           1, dilate,
                                           groups, nobias=True,
                                           initialW=initialW,
                                           bn_kwargs=bn_kwargs)
            else:
                self.conv2 = Conv2DBNActiv(mid_channels, mid_channels,
                                           3, second_stride, dilate,
                                           scale, dilate,
                                           groups, nobias=True,
                                           initialW=initialW,
                                           bn_kwargs=bn_kwargs)

            self.conv3 = Conv2DBNActiv(mid_channels, out_channels, 1, 1, 0,
                                       nobias=True, initialW=initialW,
                                       activ=None, bn_kwargs=bn_kwargs)
            if add_block == 'se':
                self.se = SEBlock(out_channels)
            elif add_block == 'gc':
                self.gc = GCBlock(out_channels)
            elif add_block is not None:
                raise ValueError
            if residual_conv:
                self.residual_conv = Conv2DBNActiv(
                    in_channels, out_channels, 1, stride, 0,
                    nobias=True, initialW=initialW,
                    activ=None, bn_kwargs=bn_kwargs)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        if hasattr(self, 'se'):
            h = self.se(h)
        elif hasattr(self, 'gc'):
            h = self.gc(h)

        if hasattr(self, 'residual_conv'):
            residual = self.residual_conv(x)
        else:
            residual = x
        h += residual
        h = F.relu(h)
        return h
