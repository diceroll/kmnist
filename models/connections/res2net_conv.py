import chainer
from chainer.functions import split_axis, concat
from chainer.links import Convolution2D


class Res2NetConv(chainer.Chain):

    def __init__(self, in_channels, out_channels,
                 ksize, stride, pad, scale, nobias=True,
                 initialW=None, initial_bias=None,
                 dilate=1, groups=1):

        assert scale > 1
        assert in_channels % scale == 0
        self.scale = scale
        super(Res2NetConv, self).__init__()

        k_ch = in_channels // scale
        with self.init_scope():
            for i in range(2, scale + 1):
                name = 'k{}'.format(i)
                conv = Convolution2D(
                    k_ch, k_ch, ksize, stride, pad,
                    nobias, initialW, initial_bias,
                    dilate=dilate, groups=groups)
                self.add_link(name, conv)

    def forward(self, x):
        x = split_axis(x, self.scale, 1)

        y = [x[0]]
        y.append(self.k2(x[1]))

        for i in range(3, self.scale + 1):
            h = x[i - 1] + y[i - 2]
            y.append(self['k{}'.format(i)](h))

        return concat(y)
