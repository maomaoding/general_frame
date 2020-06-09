import os,time,shutil
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
				 base_size=520, crop_size=480, **kwargs):
		super(curbstonedata, self).__init__()
		self.train_list = train_list
		self.val_list = val_list
		self.mode = mode
		self.transform = transform
		self.base_size = base_size
		self.crop_size = crop_size
		self.images, self.mask_paths = self._get_data_pairs()
		assert (len(self.images) == len(self.mask_paths))
		if len(self.images) == 0:
			raise RuntimeError("len(self.images) == 0" + "\n")

	def __getitem__(self, index):
		img = Image.open(self.images[index]).convert('RGB')
		mask = Image.open(self.mask_paths[index])

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
		outsize = self.crop_size
		short_size = outsize
		w, h = img.size
		if w > h:
			oh = short_size
			ow = int(1.0 * w * oh / h)
		else:
			ow = short_size
			oh = int(1.0 * h * ow / w)
		img = img.resize((ow, oh), Image.BILINEAR)
		mask = mask.resize((ow, oh), Image.NEAREST)
		# center crop
		w, h = img.size
		x1 = int(round((w - outsize) / 2.))
		y1 = int(round((h - outsize) / 2.))
		img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
		mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
		# final transform
		img, mask = self._img_transform(img), self._mask_transform(mask)
		return img, mask

	def _sync_transform(self, img, mask):
		crop_size = self.crop_size
		# random scale (short edge)
		short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
		w, h = img.size
		if h > w:
			ow = short_size
			oh = int(1.0 * h * ow / w)
		else:
			oh = short_size
			ow = int(1.0 * w * oh / h)
		img = img.resize((ow, oh), Image.BILINEAR)
		mask = mask.resize((ow, oh), Image.NEAREST)
		# pad crop
		if short_size < crop_size:
			padh = crop_size - oh if oh < crop_size else 0
			padw = crop_size - ow if ow < crop_size else 0
			img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
			mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
		# random crop crop_size
		w, h = img.size
		x1 = random.randint(0, w - crop_size)
		y1 = random.randint(0, h - crop_size)
		img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
		mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
		# gaussian blur as in PSP
		if random.random() < 0.5:
			img = img.filter(ImageFilter.GaussianBlur(
				radius=random.random()))
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
	dataset = curbstonedata(mode='val', transform=input_transform)
	img, label = dataset[1]
	print(img.size())