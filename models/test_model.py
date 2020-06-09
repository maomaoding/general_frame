import sys
sys.path.append('..')
from models import create_model
from config import opts
import torch

opt = opts.opts()
opt.from_file('./config/configs/efficientdet.py')

net = create_model(opt)
input = torch.randn(4,3,512,512)
features, regression, classification, anchors = net(input)
print(regression.size(), classification.size(), anchors.size())