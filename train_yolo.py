import find_mxnet
import mxnet as mx
import logging
import os


def get_model(num_classes=1470):
    """
    Yon only look once: Unified, Real-Time Object Detection
    """
    data = mx.symbol.Variable('data')
    # 1st layer
    conv1 = mx.symbol.Convolution(data=data, kernal=(7,7),
                                  num_filter=64, stride=(2,2), pad=(1,1))
    relu1 = mx.symbol.LeakyReLU(data=conv1, slope=0.1)
    pool1 = mx.symbol.Pooling(data=relu1, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 2nd layer
    conv2 = mx.symbol.Convolution(data=pool1, kernal=(3,3),
                                  num_filter=192, stride=(1,1), pad=(1,1))
    relu2 = mx.symbol.LeakyRelu(data=conv2, slope=0.1)
    pool2 = mx.symbol.Pooling(data=relu2, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 3rd layer
    conv3_1 = mx.symbol.Convolution(data=pool2, kernal=(1,1),
                                    num_filter=128, stride=(1,1), pad=(1,1))
    relu3_1 = mx.symbol.LeakyRelu(data=conv3_1, slope=0.1)
    conv3_2 = mx.symbol.Convolution(data=relu3_1, kernal=(3,3),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu3_2 = mx.symbol.LeakyRelu(data=conv3_2, slope=0.1)
    conv3_3 = mx.symbol.Convolution(data=relu3_2, kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu3_3 = mx.symbol.LeakyRelu(data=conv3_3, slope=0.1)
    conv3_4 = mx.symbol.Convolution(data=relu3_3, kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu3_4 = mx.symbol.LeakyRelu(data=relu3_4, slope=0.1)
    pool3 = mx.symbol.Pooling(data=relu3_4, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 4th layer
    conv4_1 = mx.symbol.Convolution(data=pool3, kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_1 = mx.symbol.LeakyRelu(data=conv4_1, slope=0.1)
    conv4_2 = mx.symbol.Convolution(data=relu4_1, kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_2 = mx.symbol.LeakyRelu(data=conv4_2, slope=0.1)
    conv4_3 = mx.symbol.Convolution(data=relu4_2, kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_3 = mx.symbol.LeakyRelu(data=conv4_3, slope=0.1)
    conv4_4 = mx.symbol.Convolution(data=relu4_3, kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_4 = mx.symbol.LeakyRelu(data=conv4_4, slope=0.1)
    conv4_5 = mx.symbol.Convolution(data=relu4_4, kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_5 = mx.symbol.LeakyRelu(data=conv4_5, slope=0.1)
    conv4_6 = mx.symbol.Convolution(data=relu4_5, kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_6 = mx.symbol.LeakyRelu(data=conv4_6, slope=0.1)
    conv4_7 = mx.symbol.Convolution(data=relu4_6, kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_7 = mx.symbol.LeakyRelu(data=conv4_7, slope=0.1)
    conv4_8 = mx.symbol.Convolution(data=relu4_7, kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_8 = mx.symbol.LeakyRelu(data=conv4_8, slope=0.1)
    conv4_9 = mx.symbol.Convolution(data=relu4_8, kernal=(1,1),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_9 = mx.symbol.LeakyRelu(data=conv4_9, slope=0.1)
    conv4_10 = mx.symbol.Convolution(data=relu4_9, kernal=(3,3),
                                     num_filter=1024, stride=(1,1), pad=(1,1))
    relu4_10 = mx.symbol.LeakyRelu(data=conv4_10, slope=0.1)
    pool4 = mx.symbol.Pooling(data=relu4_10, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 5th layer
    conv5_1 = mx.symbol.Convolution(data=pool4, kernal=(1,1),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu5_1 = mx.symbol.LeakyRelu(data=conv5_1, slope=0.1)
    conv5_2 = mx.symbol.Convolution(data=relu5_1, kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu5_2 = mx.symbol.LeakyRelu(data=conv5_2, slope=0.1)
    conv5_3 = mx.symbol.Convolution(data=relu5_2, kernal=(1,1),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu5_3 = mx.symbol.LeakyRelu(data=conv5_3, slope=0.1)
    conv5_4 = mx.symbol.Convolution(data=relu5_3, kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu5_4 = mx.symbol.LeakyRelu(data=conv5_4, slope=0.1)
    conv5_5 = mx.symbol.Convolution(data=relu5_4, kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu5_5 = mx.symbol.LeakyRelu(data=conv5_5, slope=0.1)
    conv5_6 = mx.symbol.Convolution(data=relu5_5, kernal=(3,3),
                                    num_filter=1024, stride=(2,2), pad=(1,1))
    relu5_6 = mx.symbol.LeakyRelu(data=conv5_6, slope=0.1)
    # 6th layer
    conv6_1 = mx.symbol.Convolution(data=relu5_6, kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu6_1 = mx.symbol.LeakyRelu(data=conv6_1, slope=0.1)
    conv6_2 = mx.symbol.Convolution(data=relu6_1, kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu6_2 = mx.symbol.LeakyRelu(data=conv6_2, slope=0.1)
    # 7th layer
    fc7 = mx.symbol.FullyConnected(data=relu6_2, num_hidden=4096)
    relu7 = mx.symbol.LeakyRelu(data=fc7, slope=0.1)
    # 8th layer
    fc8 = mx.symbol.FullyConnected(data=relu7, num_hidden=num_classes)
    # linear
    output = mx.symbol.LeakyRelu(data=fc8, slope=1)
    return output
