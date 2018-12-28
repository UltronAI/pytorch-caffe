"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""

import os
import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torchvision

from ConvertModel import ConvertModel_caffe
from ConvertModel import ConvertModel_ncnn
from ReplaceDenormals import ReplaceDenormals

import argparse

parser = argparse.ArgumentParser(description='Convert Pytorch model to Caffe or NCNN model.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--target-type', type=str, choices=['caffe'], default='caffe')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--model-type', type=str, choices=['dispnet', 'posenet', 'stereo'], default='dispnet')
parser.add_argument('--output-dir', type=str, default='output')
parser.add_argument('-g', '--gpu-id', type=int, metavar='N', default=1)

args = parser.parse_args()
print args
""" Import your net structure here """

from models import QuantDispNetS, QuantPoseExpNet, OdometryNet

#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

def change_state_dict_keyname(weights):
    from collections import OrderedDict
    new_weights = OrderedDict()
    
    for key, value in weights.items():
        lname = key.split('.')[0]
        if lname == 'fc_pose':
            new_weights[key] = value
        else:
            new_key = lname + '.0.' + key.split('.')[-1]
            new_weights[new_key] = value
    return new_weights

def main():

    """  Init pytorch model  """
    print 'Creating model...'
    weights = torch.load(args.model)
    if args.model_type == 'dispnet':
        model =QuantDispNetS()
        input_shape = (1, 3, 128, 416)
    elif args.model_type == 'posenet':
        seq_length = int(weights['state_dict']['conv1_1.weight'].size(1)/3)
        model = QuantPoseExpNet(nb_ref_imgs=seq_length-1, output_exp=True)
        # input_shape = [(1, 3, 128, 416)] * seq_length
        input_shape = (1, 3 * seq_length, 128, 416)
    elif args.model_type == 'stereo':
        model = OdometryNet()
        input_shape = (1, 6, 160, 608)
    else:
        print 'Invalid Model: {}'.format(args.model_type)
        exit(0)

    print 'Loading weights...'
    if args.model_type == 'dispnet' or args.model_type == 'posenet':
        model.load_state_dict(weights['state_dict'])
    else:
        model.load_state_dict(change_state_dict_keyname(weights))

    """ Replace denormal weight values(<1e-30), otherwise may increase forward time cost """
    ReplaceDenormals(model)

    """  Connnnnnnnvert!  """
    print 'Converting...'
    text_net, binary_weights = ConvertModel_caffe(model, input_shape, args.model_type, softmax=False)

    """  Save files  """
    save_path = os.path.join(args.output_dir, args.model_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print 'Saving to ' + save_path

    import google.protobuf.text_format
    with open(save_path + '/' + args.model_type + '.prototxt', 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(text_net))
    with open(save_path + '/' + args.model_type + '.caffemodel', 'w') as f:
        f.write(binary_weights.SerializeToString())

    print 'Converting Done.'


if __name__ == '__main__':
    main()
