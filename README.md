Using [MXNet](https://github.com/dmlc/mxnet) for Object Detection
===
Recent object detection approaches based deep learning use image classification task to pretrain a convolutional neural network model. This repository contains examples for Image Classification and Object Detection.

Install MXNet
---
[Install Guide](http://mxnet.readthedocs.io/en/latest/how_to/build.html) for MXNet.

Image Classification
---
We train the small [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) for Image Classification using [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which consists of 60000 32x32 color images in 10 classes. The small AlexNet is a four-layer CNN. The training image's shape is (3, 24, 24) with random cropping and mirroring. 
You can find the layer definition [here](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-conv-local-13pct.cfg) and the layer parameter [here](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-conv-local-13pct.cfg). AlexNet symbol definition is in symbol_alexnet.py.
Here is a [Classification datasets results board](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130).

Object Detection
---
[YOLO](http://pjreddie.com/darknet/yolo/) is a new real-time approach to object detection.

Reference
---
[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems. 2012.   
[2] Redmon, Joseph, et al. You only look once: Unified, real-time object detection. arXiv preprint arXiv:1506.02640. 2015.
