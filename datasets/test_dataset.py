import sys
sys.path.append('..')
from datasets import *
from config import opts
import torch,os
import numpy as np
from utils.visualizer import Visualizer

opt = opts.opts()
opt.from_file('./config/configs/centernet.py')
debugger = Visualizer(opt)

test_dataset = get_dataset(opt, 'train')
# loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=8,
# 									collate_fn=test_dataset.collate_fn)
loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True, num_workers=1)
index = 3
for iter_id, batch in enumerate(loader):
	img = batch['input'][index].detach().cpu().numpy().transpose(1, 2, 0)
	img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
	# dets_gt = batch['gt'].numpy().reshape(dets.shape[0], -1, dets.shape[2])
	dets_gt = batch['gt']
	dets_gt[:, :, :4] *= opt.down_ratio

	gt = debugger.gen_colormap(batch['hm'][index].detach().cpu().numpy())

	debugger.add_blend_img(img, gt, 'gt_hm')
	debugger.add_img(img, img_id='out_gt')
	for k in range(len(dets_gt[0])):
		if dets_gt[index, k, 4] > 0:
			debugger.add_coco_bbox(dets_gt[index, k, :4], dets_gt[index, k, -1],
								   dets_gt[index, k, 4],
								   img_id='out_gt')
	debugger.show_all_imgs(pause=True)

	os._exit(0)