import torch.utils.data as data
import cv2,os
import numpy as np
from datasets.augment import *

class BaseTask(data.Dataset):
	def __init__(self):
		super(BaseTask, self).__init__()

	def __getitem__(self, index):
		# try:
			img_path = self.get_img_path(index)
			ann_path = self.get_ann(index)

			img, ann = self.prepare_input(img_path, ann_path)

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

	def collate_fn(self, data):
		raise NotImplementedError