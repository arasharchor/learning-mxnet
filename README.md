# Learning MXNet
> [MXNet](https://github.com/dmlc/mxnet) is a deep learning framework. This repository contains examples for Image Classification and Object Detection.

## Install MXNet
Install MXNet by [Guide](http://mxnet.readthedocs.io/en/latest/how_to/build.html).

## Image Classification
Recent object detection approaches based deep learning use Image Classification task to pretrain a convolutional neural network model. We train a small AlexNet([def](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-conv-local-13pct.cfg), [params](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-conv-local-13pct.cfg), [ref](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)) for Image Classification using [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which consists of 60000 32x32 color images in 10 classes.

Here is a [Classification datasets results board](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).

## Object Detection
[Yon only look once: Unified, Real-Time Object Detection](http://pjreddie.com/darknet/yolo/) (YOLO) is a new real-time approach to object detection.
