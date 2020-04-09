import os,time,shutil
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import random
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import lovasz_losses as L

from PIL import Image, ImageOps, ImageFilter
from torch.autograd import Variable

class curbstonedata(data.Dataset):
	def __init__(self, train_list='/home/cowa/data_server/dataset/COWA/lane/dyh/train.txt',
						val_list='/home/cowa/data_server/dataset/COWA/lane/dyh/val.txt',
						mode=None, transform=None,
						crop_size=(256,512), **kwargs):
		super(curbstonedata, self).__init__()
		self.train_list = train_list
		self.val_list = val_list
		self.mode = mode
		self.transform = transform
		self.crop_size = crop_size
		self.images, self.mask_paths = self._get_data_pairs()
		assert (len(self.images) == len(self.mask_paths))
		if len(self.images) == 0:
			raise RuntimeError("len(self.images) == 0" + "\n")

	def __getitem__(self, index):
		img = cv2.imread(self.images[index])
		mask = cv2.imread(self.mask_paths[index],0)

		# synchrosized transform
		if self.mode == 'train':
			img, mask = self._sync_transform(img, mask)
		else:
			img, mask = self._val_sync_transform(img, mask)

		# general resize, normalize and toTensor
		if self.transform is not None:
			img = self.transform(img)
		return img, mask

	def _val_sync_transform(self, img, mask):
		ow = self.crop_size[1]
		oh = self.crop_size[0]
		img = cv2.resize(img, (ow, oh), Image.BILINEAR)
		mask = cv2.resize(mask, (ow, oh), Image.NEAREST)
		# final transform
		img, mask = self._img_transform(img), self._mask_transform(mask)
		return img, mask

	def _sync_transform(self, img, mask):
		# gaussian blur
		if random.random() < 0.5:
			img = cv2.GaussianBlur(img, (5,5), random.random()+1)

		#affine augment
		ratio = random.random() - 0.5
		pts1 = np.float32([[0, 0], [0, img.shape[0]-1], [img.shape[1]-1, img.shape[0]-1]])
		pts2 = np.float32([[ratio*img.shape[0], 0], [0, img.shape[0]-1], [(1-ratio)*(img.shape[1]-1), img.shape[0]-1]])
		M = cv2.getAffineTransform(pts1, pts2)
		img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
		mask = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]))

		ori_width = img.shape[1]
		ori_height = img.shape[0]
		random_scale = float(random.randint(8, 12)) / 10.
		scale_width = int(ori_width * random_scale)
		scale_height = int(ori_height * random_scale)
		if scale_width > ori_width:
			padw = scale_width - ori_width
			padh = scale_height - ori_height
			img = cv2.copyMakeBorder(img, int(padh/2), padh-int(padh/2), int(padw/2), padw-int(padw/2),
									cv2.BORDER_CONSTANT, value=0)
			mask = cv2.copyMakeBorder(mask, int(padh/2), padh-int(padh/2), int(padw/2), padw-int(padw/2),
									cv2.BORDER_CONSTANT, value=0)
		else:
			x1 = random.randint(0, ori_width - scale_width)
			y1 = random.randint(0, ori_height - scale_height)
			img = img[y1:y1 + scale_height, x1:x1 + scale_width]
			mask = mask[y1:y1 + scale_height, x1:x1 + scale_width]

		img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
		mask = cv2.resize(mask, (self.crop_size[1], self.crop_size[0]), Image.NEAREST)
		# final transform
		img, mask = self._img_transform(img), self._mask_transform(mask)
		return img, mask

	def _img_transform(self, img):
		return np.array(img)

	def _mask_transform(self, mask):
		return torch.LongTensor(np.array(mask).astype('int32'))

	def __len__(self):
		return len(self.images)

	def _get_data_pairs(self):
		root = '/home/cowa/data_server/dataset/COWA/lane/dyh'
		img_paths = []
		mask_paths = []
		if self.mode == 'train':
			with open(self.train_list, 'r') as f:
				for line in f:
					line = line.strip('\r\n')
					img_paths.append(os.path.join(root, line.split(' ')[0]))
					mask_paths.append(os.path.join(root, line.split(' ')[1]))
		else:
			with open(self.val_list, 'r') as f:
				for line in f:
					line = line.strip('\r\n')
					img_paths.append(os.path.join(root, line.split(' ')[0]))
					mask_paths.append(os.path.join(root, line.split(' ')[1]))
		return img_paths, mask_paths

if __name__ == '__main__':
	input_transform = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
	])
	dataset = curbstonedata(mode='train', transform=input_transform)
	img, label = dataset[1]
	print(img.size())
	print(label.size())