import argparse
import random
from pathlib import Path

import chainer
import chainer.functions as F
import cupy
import numpy as np
import pandas as pd
from chainer import function, iterators, serializers
from chainer.training import extensions
from tqdm import tqdm, trange

from dataloader import KMNIST
from models.modified_classifier import ModifiedClassifier
from models.se_res2net import SERes2Net50
from models.se_resnext import SEResNeXt50, SEResNeXt101


def main():
    parser = argparse.ArgumentParser(description='training mnist')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--n_tta', type=int, default=1,
                        help='Number of test time augmentations')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--save_prob', action='store_true',
                        help='Save model prediction probabilities')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    cupy.random.seed(args.seed)
    chainer.config.cudnn_deterministic = True

    resize = [('Resize', {'p': 1.0, 'height': 32, 'width': 32})]

    models = [
        ('SEResNeXt101', None,
         ['output/se_resnext101_28_fold_1/best_model.npz',
          'output/se_resnext101_28_fold_2/best_model.npz',
          'output/se_resnext101_28_fold_3/best_model.npz',
          'output/se_resnext101_28_fold_4/best_model.npz',
          'output/se_resnext101_28_fold_5/best_model.npz']),
        ('SEResNeXt101', resize,
         ['output/se_resnext101_32_fold_1/best_model.npz',
          'output/se_resnext101_32_fold_2/best_model.npz',
          'output/se_resnext101_32_fold_3/best_model.npz',
          'output/se_resnext101_32_fold_4/best_model.npz',
          'output/se_resnext101_32_fold_5/best_model.npz']),
    ]

    xp = cupy if args.gpu >= 0 else np
    batchsize = args.batchsize
    n_tta = args.n_tta
    preds = []
    pbar1 = tqdm(models, ncols=100, leave=False)

    for model_name, augmentation, model_pathes in pbar1:
        pbar1.set_postfix(model_name=model_name)
        test_data = KMNIST(augmentation=augmentation, train=False)
        test_iter = iterators.SerialIterator(test_data, batchsize,
                                             repeat=False, shuffle=False)
        n_data = len(test_data)
        model = ModifiedClassifier(eval(model_name)())

        pbar2 = tqdm(model_pathes, ncols=100, leave=False)
        for model_path in pbar2:
            pbar2.set_postfix(model_path=model_path)
            serializers.load_npz(model_path, model)
            if args.gpu >= 0:
                chainer.backends.cuda.get_device_from_id(args.gpu).use()
                model.to_gpu()

            pbar3 = trange(n_tta, ncols=100, leave=False)
            for i in pbar3:
                pbar3.set_postfix(tta=i+1)
                test_iter.reset()

                pred = []
                with function.no_backprop_mode(), chainer.using_config(
                        'train', False):
                    pbar4 = trange(
                        n_data // batchsize + (n_data % batchsize != 0),
                        ncols=100, leave=False)
                    for j in pbar4:
                        batch = test_iter.next()
                        pbar4.set_postfix(data=batchsize * (j + 1))
                        x = xp.array(batch)
                        y = F.softmax(model.predictor(x))
                        pred.extend(chainer.backends.cuda.to_cpu(y.data))
                    pbar4.close()
                preds.append(np.array(pred))
            pbar3.close()
        pbar2.close()
    pbar1.close()

    preds = np.array(preds)
    if preds.ndim > 2:
        preds = np.mean(preds, axis=0)

    result = pd.DataFrame(columns=['ImageId', 'Label'])
    result.ImageId = np.arange(len(test_data)) + 1
    result.Label = np.argmax(preds, axis=1)
    if args.save_prob:
        result['Prob'] = np.max(preds, axis=1)
    result.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
