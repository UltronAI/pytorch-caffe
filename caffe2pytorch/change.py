import torch
from collections import OrderedDict

ori_path = './temporal-caffemodel.pth.tar'
#new_path = './odometry_caffemodel.pth.tar'

state_dict = torch.load(ori_path)
new_state_dict = OrderedDict()

for key, value in state_dict.items():
    lname = key.split('.')[0]
    if lname.split('_')[0] == 'conv':
        new_lname = 'conv' + str(int(lname.split('_')[1])+1)
        new_key = new_lname + '.' + key.split('.')[-1]
        new_state_dict[new_key] = value
    elif lname.split('_')[0] == 'fc':
        new_lname = 'fc' + str(int(lname.split('_')[1])+1)
        new_key = new_lname + '.' + key.split('.')[-1]
        new_state_dict[new_key] = value
    elif lname.split('_')[0] == 'temporal':
        new_lname = 'fc_pose'
        new_key = new_lname + '.' + key.split('.')[-1]
        new_state_dict[new_key] = value

torch.save(new_state_dict, 'temporal.pth.tar')
