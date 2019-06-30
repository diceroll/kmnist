import numpy as np

import chainer
import chainer.functions as F
from chainer.links import Convolution2D


class AugmentedConv(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize,
                 dk, dv, Nh, relative, initialW=None):

        super(AugmentedConv, self).__init__()

        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.relative = relative

        with self.init_scope():
            self.conv = Convolution2D(in_channels, out_channels - dv, ksize,
                                      stride=1, pad=ksize//2,
                                      nobias=True, initialW=initialW)
            self.conv_qkv = Convolution2D(in_channels, 2 * dk + dv, ksize,
                                          stride=1, pad=ksize//2,
                                          nobias=True, initialW=initialW)
            self.conv_attn = Convolution2D(dv, dv, 1,
                                           nobias=True, initialW=initialW)

    def forward(self, x):
        B, C, H, W = x.shape

        conv_out = self.conv(x)

        flat_q, flat_k, flat_v = self.compute_flat_qkv(x, self.dk,
                                                       self.dv, self.Nh)

        logits = F.matmul(flat_q.transpose((0, 1, 3, 2)), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(
                flat_q.reshape((B, self.Nh, self.dk // self.Nh, H, W)))
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, axis=1)

        attn_out = F.matmul(weights, flat_v.transpose((0, 1, 3, 2)))
        attn_out = attn_out.reshape((B, self.Nh, self.dv // self.Nh, H, W))
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.conv_attn(attn_out)
        print(conv_out.shape, attn_out.shape)

        return F.concat((conv_out, attn_out), axis=1)

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.shape

        xp = chainer.backend.get_array_module(x)
        col_pad = xp.zeros((B, Nh, L, 1), xp.float32)
        x = F.concat((x, col_pad), axis=3)
        flat_x = x.reshape((B, Nh, L * 2 * L))
        flat_pad = xp.zeros((B, Nh, L - 1), xp.float32)
        flat_x_padded = F.concat((flat_x, flat_pad), axis=2)

        final_x = flat_x_padded.reshape((B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L-1:]

        return final_x

    def relative_logits_1d(self, q, rel_k, H, W, Nh, transpose_mask):
        rel_logits = F.einsum('bhxyd,md->bhxym', q, rel_k)

        rel_logits = rel_logits.reshape((-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = rel_logits.reshape((-1, Nh, H, W, W))
        rel_logits = F.expand_dims(rel_logits, axis=3)
        rel_logits = F.tile(rel_logits, (1, 1, 1, H, 1, 1))

        rel_logits = rel_logits.transpose(transpose_mask)
        rel_logits = rel_logits.reshape((-1, Nh, H * W, H * W))

        return rel_logits

    def relative_logits(self, q):
        B, Nh, dkh, H, W = q.shape
        q = q.transpose((0, 1, 3, 4, 2))

        xp = chainer.backend.get_array_module(q)
        key_rel_w = xp.random.randn(2 * W - 1, dkh).astype(xp.float32)
        mask_w = [0, 1, 2, 4, 3, 5]
        rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, Nh, mask_w)

        key_rel_h = xp.random.randn(2 * H - 1, dkh).astype(xp.float32)
        mask_h = [0, 1, 4, 2, 5, 3]
        rel_logits_h = self.relative_logits_1d(q.transpose((0, 1, 3, 2, 4)),
                                               key_rel_h, W, H, Nh, mask_h)

        return rel_logits_h, rel_logits_w

    def split_heads_2d(self, x, Nh):
        B, C, H, W = x.shape

        return x.reshape((B, Nh, C // Nh, H, W))

    def combine_heads_2d(self, x):
        B, Nh, dv, H, W = x.shape

        return x.reshape((B, Nh * dv, H, W))

    def compute_flat_qkv(self, x, dk, dv, Nh):
        B, _, H, W = x.shape

        qkv = self.conv_qkv(x)
        q = self.split_heads_2d(qkv[:, :dk], Nh)
        k = self.split_heads_2d(qkv[:, dk:dk * 2], Nh)
        v = self.split_heads_2d(qkv[:, dk * 2:], Nh)

        dkh = dk // Nh
        q *= dkh ** (-0.5)

        flat_q = q.reshape((B, Nh, dk // Nh, H * W))
        flat_k = k.reshape((B, Nh, dk // Nh, H * W))
        flat_v = v.reshape((B, Nh, dv // Nh, H * W))

        return flat_q, flat_k, flat_v


if __name__ == '__main__':

    x = np.ones((100, 16, 8, 8), np.float32)
    conv = AugmentedConv(16, 32, 3, dk=8, dv=8, Nh=8, relative=True)

    y = conv(x)
    print(y.shape)
