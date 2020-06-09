import os, cv2
import numpy as np
import torch.utils.data as data
from ..augment.transforms import *

class Junctions(data.Dataset):
	def get_ann(self, index):
		label_path = self.label_list[index]
		labels = []
		for bb in open(label_path, 'r'):
			xmin, ymin, xmax, ymax, classid = bb.rstrip().split()
			label = [float(xmin), float(ymin), float(xmax), float(ymax), float(classid)]
			labels.append(label)
		return labels

	def get_img_path(self, index):
		return self.img_list[index]

	def prepare_input(self, img_path, anns):
		input_img = cv2.imread(img_path)
		height, width = input_img.shape[0], input_img.shape[1]
		if self.opt.keep_res:
			input_h = ((height - 1) | self.opt.pad) + 1  # 获取大于height并能被self.opt.pad整除的最小整数
			input_w = ((width - 1) | self.opt.pad) + 1
		else:
			input_h = ((self.opt.input_h-1)|self.opt.pad) + 1
			input_w = ((self.opt.input_w-1)|self.opt.pad) + 1
		anns = np.array(anns)
		for ann in anns:
			ann[[0, 2]] = ann[[0, 2]] / input_img.shape[1] * input_w
			ann[[1, 3]] = ann[[1, 3]] / input_img.shape[0] * input_h
		input_img = cv2.resize(input_img, (input_w, input_h), interpolation=cv2.INTER_AREA)

		#for data augmentation
		for item in self.augment_list:
			input_img, anns = item(input_img, anns)

		return input_img, anns

	def __init__(self, opt, split='train'):
		super(Junctions, self).__init__()
		self.data_dir = os.path.join(opt.data_dir, opt.dataset)
		self.train_root = os.path.join(self.data_dir, split)

		self.opt = opt
		self.split = split
		if split == 'train':
			self.augment_list = [RandomCrop(0.5,0.1), RandomGaussBlur(0.3), RandomContrastBright(0.3), RandomNoise(0.3),
								Normalize(opt.mean, opt.std)]
		else:
			self.augment_list = [Normalize(opt.mean, opt.std)]

		print('==> initializing junctions {} data.'.format(split))
		self.img_list = []
		self.label_list = []
		for path in open(os.path.join(self.data_dir, split+'.txt'), 'r'):
			img, label = path.rstrip().split(' ')
			self.img_list.append(os.path.join(self.train_root, img))
			self.label_list.append(os.path.join(self.train_root, label))
		self.num_samples = len(self.img_list)
		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples