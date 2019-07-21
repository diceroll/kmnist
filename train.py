import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path

import chainer
import cupy
import numpy as np
from chainer import iterators, optimizers, serializers
from chainer.training import StandardUpdater, Trainer, extensions, triggers

from cosine_annealing import CosineAnnealing
from dataloader import KMNIST
from models.modified_classifier import ModifiedClassifier
from models.se_res2net import SERes2Net50
from models.se_resnext import SEResNeXt50, SEResNeXt101


def main():
    parser = argparse.ArgumentParser(description='training mnist')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--n_fold', '-nf', type=int, default=5,
                        help='n_fold cross validation')
    parser.add_argument('--fold', '-f', type=int, default=1)
    parser.add_argument('--out_dir_name', '-dn', type=str, default=None,
                        help='Name of the output directory')
    parser.add_argument('--report_trigger', '-rt', type=str, default='1e',
                        help='Interval for reporting(Ex.100i, default:1e)')
    parser.add_argument('--save_trigger', '-st', type=str, default='1e',
                        help='Interval for saving the model'
                             '(Ex.100i, default:1e)')
    parser.add_argument('--load_model', '-lm', type=str, default=None,
                        help='Path of the model object to load')
    parser.add_argument('--load_optimizer', '-lo', type=str, default=None,
                        help='Path of the optimizer object to load')
    args = parser.parse_args()

    if args.out_dir_name is None:
        start_time = datetime.now()
        out_dir = Path('output/{}'.format(start_time.strftime('%Y%m%d_%H%M')))
    else:
        out_dir = Path('output/{}'.format(args.out_dir_name))

    random.seed(args.seed)
    np.random.seed(args.seed)
    cupy.random.seed(args.seed)
    chainer.config.cudnn_deterministic = True

    # model = ModifiedClassifier(SEResNeXt50())
    # model = ModifiedClassifier(SERes2Net50())
    model = ModifiedClassifier(SEResNeXt101())

    if args.load_model is not None:
        serializers.load_npz(args.load_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(1e-4))
    if args.load_optimizer is not None:
        serializers.load_npz(args.load_optimizer, optimizer)

    n_fold = args.n_fold
    slices = [slice(i, None, n_fold) for i in range(n_fold)]
    fold = args.fold - 1

    # model1
    # augmentation = [
    #     ('Rotate', {'p': 0.8, 'limit': 5}),
    #     ('PadIfNeeded', {'p': 0.5, 'min_height': 28, 'min_width': 30}),
    #     ('PadIfNeeded', {'p': 0.5, 'min_height': 30, 'min_width': 28}),
    #     ('Resize', {'p': 1.0, 'height': 28, 'width': 28}),
    #     ('RandomScale', {'p': 1.0, 'scale_limit': 0.1}),
    #     ('PadIfNeeded', {'p': 1.0, 'min_height': 32, 'min_width': 32}),
    #     ('RandomCrop', {'p': 1.0, 'height': 28, 'width': 28}),
    #     ('Mixup', {'p': 0.5}),
    #     ('Cutout', {'p': 0.5, 'num_holes': 4, 'max_h_size': 4,
    #                 'max_w_size': 4}),
    # ]
    # resize = None

    # model2
    augmentation = [
        ('Rotate', {'p': 0.8, 'limit': 5}),
        ('PadIfNeeded', {'p': 0.5, 'min_height': 28, 'min_width': 32}),
        ('PadIfNeeded', {'p': 0.5, 'min_height': 32, 'min_width': 28}),
        ('Resize', {'p': 1.0, 'height': 32, 'width': 32}),
        ('RandomScale', {'p': 1.0, 'scale_limit': 0.1}),
        ('PadIfNeeded', {'p': 1.0, 'min_height': 36, 'min_width': 36}),
        ('RandomCrop', {'p': 1.0, 'height': 32, 'width': 32}),
        ('Mixup', {'p': 0.5}),
        ('Cutout', {'p': 0.5, 'num_holes': 4, 'max_h_size': 4,
                    'max_w_size': 4}),
    ]
    resize = [('Resize', {'p': 1.0, 'height': 32, 'width': 32})]

    train_data = KMNIST(augmentation=augmentation, drop_index=slices[fold],
                        pseudo_labeling=True)
    valid_data = KMNIST(augmentation=resize, index=slices[fold])

    train_iter = iterators.SerialIterator(train_data, args.batchsize)
    valid_iter = iterators.SerialIterator(valid_data, args.batchsize,
                                          repeat=False, shuffle=False)

    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater, (args.epoch, 'epoch'), out=out_dir)

    report_trigger = (
        int(args.report_trigger[:-1]),
        'iteration' if args.report_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.LogReport(trigger=report_trigger))
    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu),
                   name='val', trigger=report_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/accuracy', 'val/main/loss',
         'val/main/accuracy', 'elapsed_time']), trigger=report_trigger)
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key=report_trigger[1],
        marker='.', file_name='loss.png', trigger=report_trigger))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key=report_trigger[1],
        marker='.', file_name='accuracy.png', trigger=report_trigger))

    save_trigger = (int(args.save_trigger[:-1]),
                    'iteration' if args.save_trigger[-1] == 'i' else 'epoch')
    trainer.extend(extensions.snapshot_object(
        model, filename='model_{0}-{{.updater.{0}}}.npz'
        .format(save_trigger[1])), trigger=save_trigger)
    trainer.extend(extensions.snapshot_object(
        optimizer, filename='optimizer_{0}-{{.updater.{0}}}.npz'
        .format(save_trigger[1])), trigger=save_trigger)
    trainer.extend(extensions.ProgressBar())
    trainer.extend(CosineAnnealing(lr_max=0.1, lr_min=1e-6, T_0=20),
                   trigger=(1, 'epoch'))

    best_model_trigger = triggers.MaxValueTrigger(
        'val/main/accuracy', trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, filename='best_model.npz'), trigger=best_model_trigger)
    trainer.extend(extensions.snapshot_object(
        optimizer, filename='best_optimizer.npz'),
        trigger=best_model_trigger)
    best_loss_model_trigger = triggers.MinValueTrigger(
        'val/main/loss', trigger=(1, 'epoch'))
    trainer.extend(
        extensions.snapshot_object(model, filename='best_loss_model.npz'),
        trigger=best_loss_model_trigger)
    trainer.extend(extensions.snapshot_object(
        optimizer, filename='best_loss_optimizer.npz'),
        trigger=best_loss_model_trigger)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    # Write parameters text
    with open(out_dir / 'train_params.txt', 'w') as f:
        f.write('model: {}\n'.format(model.predictor.__class__.__name__))
        f.write('n_epoch: {}\n'.format(args.epoch))
        f.write('batch_size: {}\n'.format(args.batchsize))
        f.write('n_data_train: {}\n'.format(len(train_data)))
        f.write('n_data_val: {}\n'.format(len(valid_data)))
        f.write('seed: {}\n'.format(args.seed))
        f.write('n_fold: {}\n'.format(args.n_fold))
        f.write('fold: {}\n'.format(args.fold))
        f.write('augmentation: \n')
        for process, param in augmentation:
            f.write('  {}: {}\n'.format(process, param))

    trainer.run()


if __name__ == '__main__':
    main()
