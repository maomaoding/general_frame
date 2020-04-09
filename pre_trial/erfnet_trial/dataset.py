import os,sys,random,torch
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import lovasz_losses as L
from utils import *

class curbstonedata(data.Dataset):
	def __init__(self, train_list='/datastore/data/dataset/COWA/lane/train.txt',
						val_list='/datastore/data/dataset/COWA/lane/val.txt',
						mode=None, crop_size=(256,512), mean=(.485, .456, .406),
						std=(.229, .224, .225), **kwargs):
		super(curbstonedata, self).__init__()
		self.train_list = train_list
		self.val_list = val_list
		self.mode = mode
		self.mean = mean
		self.std = std
		self.crop_size = crop_size
		self.images, self.mask_paths, self.lanecls_lists = self._get_data_pairs()
		assert (len(self.images) == len(self.mask_paths) == len(self.lanecls_lists))
		if len(self.images) == 0:
			raise RuntimeError("len(self.images) == 0" + "\n")

	def __getitem__(self, index):
		img = cv2.imread(self.images[index])
		mask = cv2.imread(self.mask_paths[index],0)
		lanecls = self.lanecls_lists[index]
		'''cut respective roi'''
		row = 0
		for i in range(img.shape[0]):
			if np.sum(mask[i,:]) > 0:
				row = i
				break
		img = img[max(row-20,0):,:,:]
		mask = mask[max(row-20,0):,:]

		'''do transform'''
		if self.mode == 'train':
			train_transform = transforms.Compose([
				GroupRandomCrop((self.crop_size[0], self.crop_size[1])),
				# Grouprandomaffine(),
				GroupRandomBlur(),
				GroupRandomHorizontalFlip(),
				GroupNormalize(self.mean, self.std),
			])
			img, mask, lanecls = train_transform([img, mask, lanecls])
		else:
			img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR)
			mask = cv2.resize(mask, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_NEAREST)

			val_transform = transforms.Compose([
				GroupNormalize(self.mean, self.std),
			])
			img, mask, lanecls = val_transform([img, mask, lanecls])

		return img, mask, lanecls

	def __len__(self):
		return len(self.images)

	def _get_data_pairs(self):
		'''0	1	2	3	4
		   缺失 黄实 白实 黄虚 白虚'''
		root = '/datastore/data/dataset/COWA/lane/'
		img_paths = []
		mask_paths = []
		lanecls_lists = []
		if self.mode == 'train':
			with open(self.train_list, 'r') as f:
				for line in f:
					line = line.rstrip()
					img_path, mask_path, _, ll, el, er, rr, _, _ = line.split(' ')
					img_paths.append(os.path.join(root, img_path))
					mask_paths.append(os.path.join(root, mask_path))
					lanecls_lists.append([int(ll), int(el), int(er), int(rr)])
		else:
			with open(self.val_list, 'r') as f:
				for line in f:
					line = line.rstrip()
					img_path, mask_path, _, ll, el, er, rr, _, _ = line.split(' ')
					img_paths.append(os.path.join(root, img_path))
					mask_paths.append(os.path.join(root, mask_path))
					lanecls_lists.append([int(ll), int(el), int(er), int(rr)])
		return img_paths, mask_paths, lanecls_lists


class culane(data.Dataset):
	def __init__(self, train_list='/home/cowa/data_server/dataset/CULane/train_gt.txt',
						val_list='/home/cowa/data_server/dataset/CULane/val_gt.txt',
						mode=None, crop_size=(256,512), mean=(.485, .456, .406),
						std=(.229, .224, .225), **kwargs):
		super(culane, self).__init__()
		self.train_list = train_list
		self.val_list = val_list
		self.mode = mode
		self.mean = mean
		self.std = std
		self.crop_size = crop_size
		self.images, self.mask_paths, self.lanecls_lists = self._get_data_pairs()
		assert (len(self.images) == len(self.mask_paths) == len(self.lanecls_lists))
		if len(self.images) == 0:
			raise RuntimeError("len(self.images) == 0" + "\n")

	def __getitem__(self, index):
		img = cv2.imread(self.images[index])
		mask = cv2.imread(self.mask_paths[index],0)
		lanecls = self.lanecls_lists[index]
		img = img[240:, :, :]
		mask = mask[240:, :]

		if self.mode == 'train':
			train_transform = transforms.Compose([
				GroupRandomCrop((self.crop_size[0], self.crop_size[1])),
				# Grouprandomaffine(),
				# GroupRandomBlur(),
				# GroupRandomHorizontalFlip(),
				GroupNormalize(self.mean, self.std),
			])
			img, mask, lanecls = train_transform([img, mask, lanecls])
		else:
			img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR)
			mask = cv2.resize(mask, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_NEAREST)

			val_transform = transforms.Compose([
				GroupNormalize(self.mean, self.std),
			])
			img, mask, lanecls = val_transform([img, mask, lanecls])

		return img, mask, lanecls

	def __len__(self):
		return len(self.images)

	def _get_data_pairs(self):
		'''0	1	2	3	4
		   缺失 黄实 白实 黄虚 白虚'''
		root = '/home/cowa/data_server/dataset/CULane'
		img_paths = []
		mask_paths = []
		lanecls_lists = []
		if self.mode == 'train':
			with open(self.train_list, 'r') as f:
				for line in f:
					line = line.rstrip()
					img_path, mask_path, ll, el, er, rr = line.split(' ')
					img_paths.append(os.path.join(root, img_path[1:]))
					mask_paths.append(os.path.join(root, mask_path[1:]))
					lanecls_lists.append([int(ll), int(el), int(er), int(rr)])
		else:
			with open(self.val_list, 'r') as f:
				for line in f:
					line = line.rstrip()
					img_path, mask_path, ll, el, er, rr = line.split(' ')
					img_paths.append(os.path.join(root, img_path[1:]))
					mask_paths.append(os.path.join(root, mask_path[1:]))
					lanecls_lists.append([int(ll), int(el), int(er), int(rr)])
		return img_paths, mask_paths, lanecls_lists


if __name__ == '__main__':
	dataset = culane(mode='train')
	img, label, lanecls = dataset[155]
	print(np.unique(label))
	print(img.size())
	print(label.size())
	print(lanecls)