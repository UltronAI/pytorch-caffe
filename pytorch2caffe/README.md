# Pytorch_to_Caffe
Based on https://github.com/starimeL/PytorchConverter. This repo is to convert my dispnet and posenet from pytorch to caffe.

## Attentions
  - **Use Python2.7** 

  - **Mind the difference on ceil_mode of pooling layer among Pytorch and Caffe, ncnn**
    - You can convert Pytorch models with all pooling layer's ceil_mode=True.
    - Or compile a custom version of Caffe/ncnn with floor() replaced by ceil() in pooling layer inference.

  - **Python Errors: Use Pytorch 0.2.0 Only to Convert Your Model**
    - Higher version of pytorch 0.3.0, 0.3.1, 0.4.0 seemingly have blocked third party model conversion.
    - Please note that you can still TRAIN your model on pytorch 0.3.0~0.4.0. The converter running on 0.2.0 could still load higher version models correctly.

  - **Other Python packages requirements:**
    - to Caffe: numpy, protobuf (to gen caffe proto)
    - to ncnn: numpy
    - for testing Caffe result: pycaffe, cv2

## Support
See details in ConvertLayer_caffe.py

	{
	    'data': data,
	    'Addmm': inner_product,
	    'Threshold': ty('ReLU'),
	    'ConvNd': spatial_convolution,
	    'MaxPool2d': MaxPooling,
	    'AvgPool2d': AvgPooling,
	    'Add': eltwise,
	    'Cmax': eltwise_max,
	    'BatchNorm': batchnorm,
	    'Concat': concat,
	    'Dropout': dropout,
	    'UpsamplingBilinear2d': UpsampleBilinear,
	    'MulConstant': MulConst,
	    'AddConstant': AddConst,
	    'Softmax': softmax,
	    'Sigmoid': ty('Sigmoid'),
	    'Tanh': ty('TanH'),
	    'ELU': elu,
	    'LeakyReLU': leaky_ReLU,
	    'PReLU': PReLU,
	    'Slice': Slice,
	    'View': View,
	    'Mean': Mean
	}