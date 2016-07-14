# -*- coding: utf-8 -*-
import find_mxnet
import mxnet as mx
import argparse
import logging

# 7.13
# 1.去掉一些pool
# 2.如何检测代码正确?
#   A: 计算中间变量的尺寸


def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN on cifar10')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=1,
                        help='the initial learning rate')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='the number of training epochs')
    return parser.parse_args()


def get_alexnet_small(num_classes=10):
    input_data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(data=input_data, num_filter=30,
                                  kernel=(5, 5), pad=(2, 2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type='max',
                              kernel=(2, 2), stride=(2, 2))

    conv2 = mx.symbol.Convolution(data=pool1, num_filter=30,
                                  kernel=(3, 3), pad=(1, 1))
    relu2 = mx.symbol.Activation(data=conv2, act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2, pool_type='max',
                              kernel=(2, 2), stride=(2, 2))

    conv3 = mx.symbol.Convolution(data=pool2, num_filter=30,
                                  kernel=(3, 3), pad=(1, 1))
    relu3 = mx.symbol.Activation(data=conv3, act_type='relu')
    pool3 = mx.symbol.Pooling(data=relu3, pool_type='max',
                              kernel=(2, 2), stride=(2, 2), pad=(1, 1))

    # flatten = mx.symbol.Flatten(data=pool3)
    # fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=20)
    fc1 = mx.symbol.FullyConnected(data=pool3, num_hidden=40)
    relu4 = mx.symbol.Activation(data=fc1, act_type='relu')

    fc2 = mx.symbol.FullyConnected(data=relu4, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return softmax


def get_alexnet(sum_classes=1000):
    input_data = mx.symbol.Variable('data')
    # 1 layer
    conv1 = mx.symbol.Convolution(data=input_data, num_filter=96,
                                  kernel=(11, 11), stride=(4, 4))
    relu1 = mx.symbol.Activation(data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type='max',
                              kernel=(3, 3), stride=(2, 2))
    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    # 2 layer
    conv2 = mx.symbol.Convolution(data=lrn1, num_filter=256,
                                  kernel=(5, 5), pad=(2, 2))
    relu2 = mx.symbol.Activation(data=conv2, act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2, pool_type='max',
                              kernel=(3, 3), stride=(2, 2))
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    # 3 layer
    conv3 = mx.symbol.Convolution(data=lrn2, num_filter=384,
                                  kernel=(3, 3), pad=(1, 1))
    relu3 = mx.symbol.Activation(data=conv3, act_type='relu')
    conv4 = mx.symbol.Convolution(data=relu3, num_filter=384,
                                  kernel=(3, 3), pad=(1, 1))
    relu4 = mx.symbol.Activation(data=conv4, act_type='relu')
    conv5 = mx.symbol.Convolution(data=relu4, num_filter=256,
                                  kernel=(3, 3), pad=(1, 1))
    relu5 = mx.symbol.Activation(data=conv5, act_type='relu')
    pool3 = mx.symbol.Pooling(data=relu5, pool_type='max',
                              kernel=(3, 3), stride=(2, 2))
    # 4 layer
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten)
    relu6 = mx.symbol.Activation(data=fc1, act_type='relu')
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # 5 layer
    fc2 = mx.symbol.FullyConnected(data=dropout1)
    relu7 = mx.symbol.Activation(data=fc2, act_type='relu')
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    # 6 layer
    fc3 = mx.symbol.FullyConnected(data=dropout2)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax


def get_iterator(args):
    data_shape = (3, 28, 28)

    train = mx.io.ImageRecordIter(
        path_imgrec = 'cifar10/train.rec',
        mean_img = 'cifar10/mean.bin',
        data_shape = data_shape,
        batch_size = args.batch_size,
        shuffle = True)

    test = mx.io.ImageRecordIter(
        path_imgrec = 'cifar10/test.rec',
        mean_img = 'cifar10/mean.bin',
        data_shape = data_shape,
        batch_size = args.batch_size)

    return (train, test)


def model_fit(args, network, data_loader):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    (train, test) = data_loader(args)
    epoch_size = args.num_examples / args.batch_size

    model = mx.model.FeedForward(
        symbol = network,
        num_epoch = args.num_epochs,
        learning_rate = args.lr,
        epoch_size = epoch_size)

    model.fit(
        X = train,
        eval_data = test)


if __name__ == '__main__':
    args = parse_args()
    network = get_alexnet_small()
    model_fit(args, network, get_iterator)
