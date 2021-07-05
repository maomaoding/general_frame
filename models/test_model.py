import sys
sys.path.insert(0, '..')
from models import get_model
from config import opts
import torch
from datasets.task.detr_task import nested_tensor_from_tensor_list

opt = opts.opts()
opt.from_file('./config/configs/detr.py')

net = get_model(opt)
net.cuda()
input = torch.randn(4,3,224,224).cuda()
input = nested_tensor_from_tensor_list(input)
y = net(input)
print(y.keys())