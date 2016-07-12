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
    conv1 = mx.symbol.Convolution(data=data, name='conv1', kernal=(7,7),
                                  num_filter=64, stride=(2,2), pad=(1,1))
    relu1 = mx.symbol.Activation(data=conv1, name='relu1', act_type='softrelu')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 2nd layer
    conv2 = mx.symbol.Convolution(data=pool1, name='conv2', kernal=(3,3),
                                  num_filter=192, stride=(1,1), pad=(1,1))
    relu2 = mx.symbol.Activation(data=conv2, name='relu2', act_type='softrelu')
    pool2 = mx.symbol.Pooling(data=relu2, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 3rd layer
    conv3_1 = mx.symbol.Convolution(data=pool2, name='conv3_1', kernal=(1,1),
                                    num_filter=128, stride=(1,1), pad=(1,1))
    relu3_1 = mx.symbol.Activation(data=conv3_1, name='relu3_1', act_type='softrelu')
    conv3_2 = mx.symbol.Convolution(data=relu3_1, name='conv3_2', kernal=(3,3),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu3_2 = mx.symbol.Activation(data=conv3_2, name='relu3_2', act_type='softrelu')
    conv3_3 = mx.symbol.Convolution(data=relu3_2, name='conv3_3', kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu3_3 = mx.symbol.Activation(data=conv3_3, name='relu3_3', act_type='softrelu')
    conv3_4 = mx.symbol.Convolution(data=relu3_3, name='conv3_4', kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu3_4 = mx.symbol.Activation(data=relu3_4, name='relu3_4', act_type='softrelu')
    pool3 = mx.symbol.Pooling(data=relu3_4, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 4th layer
    conv4_1 = mx.symbol.Convolution(data=pool3, name='conv4_1', kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_1 = mx.symbol.Activation(data=conv4_1, name='relu4_1', act_type='softrelu')
    conv4_2 = mx.symbol.Convolution(data=relu4_1, name='conv4_2', kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_2 = mx.symbol.Activation(data=conv4_2, name='relu4_2', act_type='softrelu')
    conv4_3 = mx.symbol.Convolution(data=relu4_2, name='conv4_3', kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_3 = mx.symbol.Activation(data=conv4_3, name='relu4_3', act_type='softrelu')
    conv4_4 = mx.symbol.Convolution(data=relu4_3, name='conv4_4', kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_4 = mx.symbol.Activation(data=conv4_4, name='relu4_4', act_type='softrelu')
    conv4_5 = mx.symbol.Convolution(data=relu4_4, name='conv4_5', kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_5 = mx.symbol.Activation(data=conv4_5, name='relu4_5', act_type='softrelu')
    conv4_6 = mx.symbol.Convolution(data=relu4_5, name='conv4_6', kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_6 = mx.symbol.Activation(data=conv4_6, name='relu4_6', act_type='softrelu')
    conv4_7 = mx.symbol.Convolution(data=relu4_6, name='conv4_7', kernal=(1,1),
                                    num_filter=256, stride=(1,1), pad=(1,1))
    relu4_7 = mx.symbol.Activation(data=conv4_7, name='relu4_7', act_type='softrelu')
    conv4_8 = mx.symbol.Convolution(data=relu4_7, name='conv4_8', kernal=(3,3),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_8 = mx.symbol.Activation(data=conv4_8, name='relu4_8', act_type='softrelu')
    conv4_9 = mx.symbol.Convolution(data=relu4_8, name='conv4_9', kernal=(1,1),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu4_9 = mx.symbol.Activation(data=conv4_9, name='relu4_9', act_type='softrelu')
    conv4_10 = mx.symbol.Convolution(data=relu4_9, name='conv4_9', kernal=(3,3),
                                     num_filter=1024, stride=(1,1), pad=(1,1))
    relu4_10 = mx.symbol.Activation(data=conv4_10, name='relu4_10', act_type='softrelu')
    pool4 = mx.symbol.Pooling(data=relu4_10, pool_type='max',
                              kernal=(2,2), stride=(2,2))
    # 5th layer
    conv5_1 = mx.symbol.Convolution(data=pool4, name='conv5_1', kernal=(1,1),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu5_1 = mx.symbol.Activation(data=conv5_1, name='relu5_1', act_type='softrelu')
    conv5_2 = mx.symbol.Convolution(data=relu5_1, name='conv5_2', kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu5_2 = mx.symbol.Activation(data=conv5_2, name='relu5_2', act_type='softrelu')
    conv5_3 = mx.symbol.Convolution(data=relu5_2, name='conv5_3', kernal=(1,1),
                                    num_filter=512, stride=(1,1), pad=(1,1))
    relu5_3 = mx.symbol.Activation(data=conv5_3, name='relu5_3', act_type='softrelu')
    conv5_4 = mx.symbol.Convolution(data=relu5_3, name='conv5_4', kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu5_4 = mx.symbol.Activation(data=conv5_4, name='relu5_4', act_type='softrelu')
    conv5_5 = mx.symbol.Convolution(data=relu5_4, name='conv5_5', kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu5_5 = mx.symbol.Activation(data=conv5_5, name='conv5_5', act_type='softrelu')
    conv5_6 = mx.symbol.Convolution(data=relu5_5, name='conv5_6', kernal=(3,3),
                                    num_filter=1024, stride=(2,2), pad=(1,1))
    relu5_6 = mx.symbol.Activation(data=conv5_6, name='conv5_6', act_type='softrelu')
    # 6th layer
    conv6_1 = mx.symbol.Convolution(data=relu5_6, name='conv6_1', kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu6_1 = mx.symbol.Activation(data=conv6_1, name='relu6_1', act_type='softrelu')
    conv6_2 = mx.symbol.Convolution(data=relu6_1, name='conv6_2', kernal=(3,3),
                                    num_filter=1024, stride=(1,1), pad=(1,1))
    relu6_2 = mx.symbol.Activation(data=conv6_2, name='relu6_2', act_type='softrelu')
    # 7th layer
    fc7_1 = mx.symbol.FullyConnected(data=relu6_2, name='fc7_1', num_hidden=4096)
    relu7_1 = mx.symbol.Activation(data=fc7_1, name='relu7_1', act_type='softrelu')
    # 8th layer
    fc8_1 = mx.symbol.FullyConnected(data=relu7_1, name='fc8_1',
                                     num_hidden=num_classes)
    # linear
    output = mx.symbol.Activation(data=fc8_1, name='relu8_1',
                                  act_type='relu')
    return output
