import sys
sys.path.append('..')
from datasets import *
from config import opts
import torch,os
import numpy as np
from utils.visualizer import Visualizer

opt = opts.opts()
opt.from_file('./config/configs/detr.py')

test_dataset = get_dataset(opt, 'train')
train_loader = torch.utils.data.DataLoader(
	test_dataset,
	batch_size=opt.batch_size,
	shuffle=False,
	num_workers=0,
	pin_memory=True,
	drop_last=True,
	collate_fn=test_dataset.collate_fn if 'collate_fn' in dir(test_dataset) else None,
)

for iter_id, batch in enumerate(train_loader):
	print(batch['annot'])
	os._exit(0)