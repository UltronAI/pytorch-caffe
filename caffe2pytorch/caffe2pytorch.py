# -*- coding: utf-8
# from pytorch2caffe import plot_graph, pytorch2caffe
import sys
sys.path.append('/home/gaof/caffe-comp/python')
import caffe
import numpy as np
import os
from caffenet import *
import argparse
import torch

def create_network(protofile, weightfile, width=608, height=160, cuda=False):
    net = CaffeNet(protofile, width=width, height=height)
    if cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.train()
    return net

################################################################################################   
parser = argparse.ArgumentParser(description='Convert Caffe model to Pytorch model.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--define', type=str, default=None)
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--output-dir', type=str, default='output')
args = parser.parse_args()

model_def = args.define
model_weights = args.weights
model_name = model_weights.split('/')[-1].split('.')[0]
model = create_network(model_def, model_weights, width=608, height=160)
# print('1')
torch.save(model.state_dict(), os.path.join(args.output_dir, model_name + '-caffemodel.pth.tar')