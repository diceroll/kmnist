import chainer
import chainer.functions as F
import chainer.links as L
import cupy as cp


class GCBlock(chainer.Chain):

    def __init__(self, n_channel, ratio=4):

        super(GCBlock, self).__init__()
        reduction_size = n_channel // ratio

        with self.init_scope():
            self.context = L.Convolution2D(n_channel, 1, ksize=1)
            self.down = L.Linear(n_channel, reduction_size)
            self.ln = L.LayerNormalization(reduction_size)
            self.up = L.Linear(reduction_size, n_channel)

    def forward(self, u):
        B, C, H, W = u.shape

        h1 = u.reshape((B, C, H * W))

        h2 = self.context(u)
        h2 = h2.reshape((B, H * W, 1))
        h2 = F.softmax(h2)

        z = F.batch_matmul(h1, h2)

        x = F.relu(self.ln(self.down(z)))
        x = self.up(x)

        x = F.broadcast_to(x, (H, W, B, C))
        x = x.transpose((2, 3, 0, 1))

        return u + x


if __name__ == '__main__':
    x = cp.zeros((100, 64, 32, 32), cp.float32)
    block = GCBlock(64, ratio=4)
    block.to_gpu(0)
    y = block(x)
    print(y)
