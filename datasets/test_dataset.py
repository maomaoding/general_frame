import sys
sys.path.append('..')
from datasets import *
from config import opts
import torch,os

opt = opts.opts()
opt.from_file('./config/configs/centernet.py')

test_dataset = get_dataset(opt, 'train')
loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=8)
for iter_id, batch in enumerate(loader):
	print(batch['input'].size())
	# os._exit(0)