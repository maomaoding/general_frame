import torch.utils.data as data
import cv2,os
import numpy as np
from datasets.augment import *

class BaseTask(data.Dataset):
	def __init__(self, opt, split):
		super(BaseTask, self).__init__()
		if split == 'train':
			auglist = []
			for augitem in opt.data_augment:
				arg = ["{}={}".format(k,v) for k,v in augitem.items() if k != 'name']
				exec("auglist.append({}({}))".format(augitem['name'], ','.join(arg)))
			if opt.task == 'centernet':
				self.aug = DataAugment(auglist)
			else:
				self.aug = DataAugment_seg(auglist)
		else:
			auglist = []
			arg = ["{}={}".format(k,v) for k,v in opt.data_augment[-1].items() if k != 'name']
			exec("auglist.append({}({}))".format(opt.data_augment[-1]['name'], ','.join(arg)))
			if opt.task == 'centernet':
				self.aug = DataAugment(auglist)
			else:
				self.aug = DataAugment_seg(auglist)

	def __getitem__(self, index):
		# try:
			img_path = self.get_img_path(index)
			ann_path = self.get_ann(index)

			img, ann = self.prepare_input(img_path, ann_path)

			img, ann = self.aug(img, ann)

			ret = self.get_data(img, ann)
			return ret
		# except:
		# 	new_index = np.random.randint(0, len(self)-1)
		# 	return self[new_index]

	def prepare_input(self, img_path, ann_path):
		raise NotImplementedError

	def get_img_path(self, detections):
		raise NotImplementedError

	def get_ann(self, detections):
		raise NotImplementedError

	def get_data(self, detections):
		raise NotImplementedError