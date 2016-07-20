import find_mxnet
import mxnet as mx


def get_symbol(num_classes=10):
    input_data = mx.symbol.Variable('data')
    # layer 1
    conv1 = mx.symbol.Convolution(data=input_data, num_filter=64,
                                  kernel=(5, 5), pad=(2, 2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1, pool_type='max',
                              kernel=(3, 3), stride=(2, 2))
    # layer 2
    conv2 = mx.symbol.Convolution(data=pool1, num_filter=64,
                                  kernel=(5, 5), pad=(2, 2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=conv2, act_type='relu')
    pool2 = mx.symbol.Pooling(data=relu2, pool_type='max',
                              kernel=(3, 3), stride=(2, 2))
    # layer 3
    conv3 = mx.symbol.Convolution(data=pool2, num_filter=32,
                                  kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=conv3, act_type='relu')
    # layer 4
    conv4 = mx.symbol.Convolution(data=relu3, num_filter=32,
                                  kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=conv4, act_type='relu')
    # layer 5
    fc = mx.symbol.FullyConnected(data=relu4, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax
