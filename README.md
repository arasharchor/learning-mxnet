#Using [MXNet](https://github.com/dmlc/mxnet) for Object Detection
Recent object detection approaches based deep learning use image classification task to pretrain a convolutional neural network model. This repository contains examples of Image Classification and Object Detection.

##Install MXNet
[Install Guide](http://mxnet.readthedocs.io/en/latest/how_to/build.html) for MXNet.

##Image Classification
We train the [small AlexNet](https://code.google.com/p/cuda-convnet/) for Image Classification using [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which consists of 60,000 32x32 color images in 10 classes. The small AlexNet is a four-layer Convolutional Neural Network. The training image's shape is (3, 24, 24) with random cropping and mirroring. You can find the layer definition [here](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-conv-local-13pct.cfg) and the layer parameter [here](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layer-params-conv-local-13pct.cfg). AlexNet symbol definition for MXNet is in [symbol_alexnet.py](https://github.com/bertjiazheng/learning-mxnet/blob/master/classification/symbol_alexnet.py). 

Here is a recipe of how we train the network.

####Stochastic Gradient Descent
1. Start by training the net for 150 epochs with learning rate = 1e-4 without step decay, momentum = 0.9 and wd = 1e-5.
2. Resume training for another 450 epochs with 0.99 step decay, momentum = 0.99 and wd = 1e-5.
Then we get 80.48% validation accuracy.

####[RMSProp](http://arxiv.org/pdf/1308.0850v5.pdf)
Train the net by RMSProp for 600 epochs with learning rate = 1e-5, gamma1 = 0.9, 0.95, 0.99, 0.999 will get 80.44%, 80.99%, 80.40%, 80.43% validation accuracy, where gamma1 is decay factor of moving average for gradient, gradient^2.

More details of results please see [classification_result.ipynb](https://github.com/bertjiazheng/learning-mxnet/blob/master/classification/classification_result.ipynb).

##Object Detection
Short for Yon only look once, [YOLO](http://pjreddie.com/darknet/yolo/) is a real-time end-to-end object detector.

#### YOLO Pipeline
1. Divide the input image into S by S grid cell. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
2. Each grid cell predicts B bounding boxes(x, y, w, h) and confidence scores(IOU) for each box. (Normalize w, h by image width and height, parametrize x, y to be offset of the grid cell, both [0, 1])
3. Each grid cell also predicts C conditional class probabilities.

##Resource
* [Classification datasets results board](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)

##Reference
1. [Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems. 2012.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
2. [Redmon, Joseph, et al. You only look once: Unified, real-time object detection. arXiv preprint arXiv:1506.02640. 2015.](http://arxiv.org/pdf/1506.02640v5.pdf)
