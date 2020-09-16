import sys
sys.path.append('..')
from models import create_model
from config import opts
import torch

opt = opts.opts()
opt.from_file('./config/configs/SAN.py')

net = create_model(opt)
input = torch.randn(4,3,224,224).cuda()
y = net(input)
print(y.size())