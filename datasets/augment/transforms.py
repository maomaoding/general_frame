
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2,os
import sys
sys.path.append('../../')
from datasets.augment.image import get_affine_transform,affine_transform

class RandomContrastBright(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, anns=None, class_name=None):
		if np.random.random()<self.random_ratio:
			alpha=np.random.random()+0.3
			beta=int((np.random.random()-0.5)*255*0.3)
			blank = np.zeros(image.shape, image.dtype)
			image = cv2.addWeighted(image, alpha, blank, 1 - alpha, beta)
			image=np.clip(image,0,255)
		return image, anns

class RandomFlip(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, bboxes=[]):
		if np.random.random()<self.random_ratio:
			image = image[:, ::-1, :]
			height,width=image.shape[:2]
			if bboxes == []:
				return image.copy()
			bboxes[:,[0, 2]] = width - bboxes[:,[2, 0]] - 1
		return image.copy()

class RandomGaussBlur(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, anns=None):
		if np.random.random()<self.random_ratio:
			kernel_size = (np.random.randint(1,3)*2+1, np.random.randint(1,3)*2+1)
			sigma = 0.8
			image = cv2.GaussianBlur(image, kernel_size, sigma)
		return image, anns

class RandomNoise(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, anns=None):
		if np.random.random()<self.random_ratio:
			noise_num = int(0.001 * image.shape[0] * image.shape[1])
			for i in range(noise_num):
				temp_x = np.random.randint(0, image.shape[0])
				temp_y = np.random.randint(0, image.shape[1])
				image[temp_x][temp_y] = 255
		return image, anns

class RandomCrop(object):
	def __init__(self,random_ratio, shift):
		self.shift = shift
		self.random_ratio=random_ratio
	def __call__(self, image, anns=None):
		if np.random.random() < self.random_ratio:
			s = np.array([image.shape[1], image.shape[0]], dtype=np.float32)

			cf = self.shift
			h_ratio = np.clip(np.random.randn() * cf, -1 * cf, 1 * cf)
			w_ratio = np.clip(np.random.randn() * cf, -1 * cf, 1 * cf)

			topleft = [max(0, -s[0]*w_ratio), max(0, -s[1]*h_ratio)]
			bottomright = [min(s[0], s[0]-s[0]*w_ratio), min(s[1], s[1]-s[1]*h_ratio)]
			c = (np.array(topleft) + np.array(bottomright)) / 2
			s[0] = s[0] - abs(s[0] * w_ratio)
			s[1] = s[1] - abs(s[1] * h_ratio)

			trans_input = get_affine_transform(c, s, 0, [image.shape[1], image.shape[0]])

			image = cv2.warpAffine(image, trans_input,
								 (image.shape[1], image.shape[0]),
								 flags=cv2.INTER_LINEAR)
			if len(anns) == 0:
				return image
			anns[:,:2] = affine_transform(anns[:,:2], trans_input)
			anns[:,2:4] = affine_transform(anns[:,2:4], trans_input)
			anns[:,[0, 2]] = np.clip(anns[:,[0, 2]], 0, image.shape[1] - 1)
			anns[:,[1, 3]] = np.clip(anns[:,[1, 3]], 0, image.shape[0] - 1)
			#remove whose area equals 0
			areas = (anns[:, 2] - anns[:, 0]) * (anns[:, 3] - anns[:, 1])
			anns = anns[np.where(areas > 0)]
		return image, anns

class Normalize(object):
	def __init__(self,mean,std):
		self.mean=np.array(mean, dtype=np.float32)
		self.std=np.array(std, dtype=np.float32)
	def __call__(self, image, anns=None):
		image = (image.astype(np.float32) / 255.)
		image = (image - self.mean) / self.std
		image = image.transpose(2, 0, 1)
		return image, anns

if __name__ == '__main__':
	img=cv2.imread("/home/dingyaohua/datasets/junctions/train/img_data/NezgU4MTUy0ac.png")
	labelpath = '/home/dingyaohua/datasets/junctions/train/label_data/NezgU4MTUy0ac.txt'
	height, width = img.shape[:2]
	scale = 6
	img = cv2.resize(img, (int(width/scale), int(height/scale)))
	anns = []
	for line in open(labelpath, 'r'):
		xmin, ymin, xmax, ymax, label = line.rstrip().split(' ')
		anns.append([int(float(xmin)/scale), int(float(ymin)/scale), int(float(xmax)/scale), int(float(ymax)/scale), int(label)])
		# anns.append([xmin, ymin, xmax, ymax, label])
	anns = np.array(anns)

	from utils.visualizer import Visualizer
	from config.opts import opts
	opt = opts()
	opt.from_file('./config/configs/centernet.py')
	aug=DataAugment([RandomRotate90(1),], opt.class_name)
	img, anns = aug(img, anns)
	anns = np.array(anns)

	debugger = Visualizer(opt)
	debugger.add_img(img, img_id='out_gt')
	for k in range(len(anns)):
		if(anns[k,4] == 3 or 1):
			debugger.add_coco_bbox(anns[k, :4], anns[k, -1],
								   1, img_id='out_gt')
	debugger.show_all_imgs(pause=True)