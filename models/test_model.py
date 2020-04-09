import sys
sys.path.append('..')
from models import create_model
from config import opts
import torch

opt = opts.opts()
opt.from_file('./config/configs/erfnet.py')

net = create_model(opt)
input = torch.randn(5,3,256,512)
output = net(input)
print(output.size())