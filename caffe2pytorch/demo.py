# -*- coding: utf-8
# from pytorch2caffe import plot_graph, pytorch2caffe
import sys
sys.path.append('/home/gaof/caffe-comp/python')
import caffe
import numpy as np
import os
from caffenet import *
import torch
from torch.autograd import Variable
import torchvision
# from netdef_resnetvlad import ResNetVlad
import torch._utils


def create_network(protofile, weightfile, width=608, height=160, cuda=False):
    net = CaffeNet(protofile, width=width, height=height)
    if cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.train()
    return net


################################################################################################   
test_mod = True

caffemodel_dir = 'demo'
input_size = (1, 3, 382, 382)

# model_def = os.path.join(caffemodel_dir, 'model.prototxt')
# model_weights = os.path.join(caffemodel_dir, 'model.caffemodel')
# input_name = 'ConvNdBackward1'
# output_name = 'AddBackward73'
# output_name = 'MaxPool2dBackward4'
# pytorch net
# model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
model_def = '/home/gaof/workspace/Depth-VO-Feat/experiments/networks/odometry_deploy.prototxt'
model_weights = '/home/gaof/workspace/Depth-VO-Feat/experiments/networks/Temporal.caffemodel' 
model = create_network(model_def, model_weights, width=608, height=160)
# print('1')
torch.save(model.state_dict(), 'temporal-caffemodel.pth.tar')
exit(0)
# # state_dict = torch.load('./student_net_paramssingle_29.pkl')
# # model_dict = model.state_dict()

# # pretrained_dict =  {k: v for k, v in state_dict.items() if k in model_dict}
# print("1.5")
# print(pretrained_dict)
# model_dict.update(pretrained_dict)

# model.load_state_dict(model_dict)
# model.eval()

# # random input
# # image = np.random.randint(0, 255, input_size)
# image = 66*np.ones(input_size)
# input_data = image.astype(np.float32)

# # pytorch forward
# input_var = Variable(torch.from_numpy(input_data))

# # test caffemodel
# # caffe.set_device(0)
# caffe.set_mode_cpu()
# net = caffe.Net(model_def, model_weights, caffe.TEST)

# net.blobs['data'].data[...] = input_data
# net.forward(start=input_name)
# caffe_output = net.blobs[output_name].data

# model = model.cpu()
# input_var = input_var.cpu()
# output_var = model(input_var)
# pytorch_output = output_var[output_name].data.cpu().numpy()

# print(input_size, pytorch_output.shape, caffe_output.shape)
# print('pytorch: min: {}, max: {}, mean: {}'.format(pytorch_output.min(), pytorch_output.max(), pytorch_output.mean()))
# print('  caffe: min: {}, max: {}, mean: {}'.format(caffe_output.min(), caffe_output.max(), caffe_output.mean()))

# diff = np.abs(pytorch_output - caffe_output)
# print('   diff: min: {}, max: {}, mean: {}, median: {}'.format(diff.min(), diff.max(), diff.mean(), np.median(diff)))


# print(pytorch_output)
