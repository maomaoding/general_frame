
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2,os
import sys
sys.path.append('../../')
from datasets.augment.image import get_affine_transform,affine_transform
def _get_border( border, size):
	i = 1
	while size - border // i <= border // i:
		i *= 2
	return border // i

#RandomCrop和RandomShiftScale只可用一个，RandomNoise只能在RandomCrop（RandomShiftScale）和RandomGasussBlur之后使用
#输入的bbox为一个二维list，第一维是目标个数，第二维是[x1,y1,x2,y2]
class DataAugment(object):
	def __init__(self, aug_list, class_name):
		self.aug_list = aug_list
		self.class_name = class_name
	def __call__(self, image, anns=None):
		# if bboxes is not None and len(bboxes)>0:
		# 	for aug in self.aug_list:
		# 		bboxes_np=np.array(bboxes)[:,:4]
		# 		extra_label=np.array(bboxes)[:,4:]
		# 		image=aug(image,bboxes_np)
		# 		bboxes_np=np.hstack([bboxes_np,extra_label])
		# 		bboxes=bboxes_np.tolist()
		# 	return image,bboxes
		# elif len(bboxes)==0:
		# 	for aug in self.aug_list:
		# 		image=aug(image)
		# 	return image, bboxes
		# else:
		# 	for aug in self.aug_list:
		# 		image=aug(image)
		# 	return image
		anns = np.array(anns)
		for aug in self.aug_list:
			image, anns = aug(image, anns, self.class_name)
		anns = anns.tolist()
		return image, anns

class DataAugment_seg(object):
	def __init__(self, aug_list):
		self.aug_list = aug_list
	def __call__(self, image, anns):
		for aug in self.aug_list:
			image, anns=aug(image,anns)
		return image,anns


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
	def __call__(self, image, anns=None, class_name=None):
		if np.random.random()<self.random_ratio:
			horizontal_or_vertical = np.random.random()
			height, width = image.shape[:2]
			if horizontal_or_vertical < 0.5: #horizontal
				image = image[:, ::-1, :]
				anns[:, [0, 2]] = width - anns[:, [2, 0]] - 1
				for i in range(len(anns)):
					old_classname = class_name[anns[i, 4]]
					if old_classname.find('right') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('right', 'left'))
					elif old_classname.find('left') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('left', 'right'))
			else:							 #vertical
				image = image[::-1, :, :]
				anns[:, [1, 3]] = height - anns[:, [3, 1]] - 1
				for i in range(len(anns)):
					old_classname = class_name[anns[i, 4]]
					if old_classname.find('top') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('top', 'bottom'))
					elif old_classname.find('bottom') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('bottom', 'top'))
		return image, anns

class RandomGaussBlur(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, anns=None, class_name=None):
		if np.random.random()<self.random_ratio:
			kernel_size = (np.random.randint(1,3)*2+1, np.random.randint(1,3)*2+1)
			sigma = 0.8
			image = cv2.GaussianBlur(image, kernel_size, sigma)
		return image, anns

class RandomNoise(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, anns=None, class_name=None):
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
	def __call__(self, image, anns=None, class_name=None):
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

loop_dict = ['top', 'right', 'bottom', 'left']
class RandomRotate90(object):
	def __init__(self, random_ratio):
		self.random_ratio = random_ratio
	def __call__(self, image, anns=None, class_name=None):
		if np.random.random() < self.random_ratio:
			height, width = image.shape[:2]
			magnitude = np.random.randint(1,4)
			matRotate = cv2.getRotationMatrix2D((width//2, height//2), -magnitude*90, 1)
			cos = np.abs(matRotate[0,0])
			sin = np.abs(matRotate[0,1])
			nW = int(height*sin + width*cos)
			nH = int(height*cos + width*sin)
			matRotate[0,2] += (nW / 2) - width//2
			matRotate[1,2] += (nH / 2) - height//2
			image = cv2.warpAffine(image, matRotate, (nW, nH))
			bboxes, label = anns[:,:4], anns[:,4]
			#change anns
			#change the label
			for i in range(len(anns)):
				old_classname = class_name[label[i]]
				if 'horizontal' in old_classname or 'vertical' in old_classname:
					if magnitude % 2 != 0:
						if old_classname.find('horizontal') != -1:
							old_classname = old_classname.replace('horizontal', 'vertical')
						else:
							old_classname = old_classname.replace('vertical', 'horizontal')
				subname_list = old_classname.split('_')
				lridx = -1
				tbidx = -1
				for j in range(len(subname_list)):
					if subname_list[j] in loop_dict:
						startidx = loop_dict.index(subname_list[j])
						endidx = (startidx + magnitude) % 4
						subname_list[j] = loop_dict[endidx]
						if subname_list[j] == 'left' or subname_list[j] == 'right':
							lridx = j
						if subname_list[j] == 'top' or subname_list[j] == 'bottom':
							tbidx = j
				if lridx != -1 and tbidx != -1 and tbidx < lridx:
					subname_list[tbidx], subname_list[lridx] = subname_list[lridx], subname_list[tbidx]
				label[i] = class_name.index('_'.join(subname_list))
			#change the bboxes
			toplefts = bboxes[:,:2]
			tmp = np.ones(len(toplefts))
			homogeneous = np.hstack([toplefts, np.expand_dims(tmp, tmp.ndim)])
			final_coord_toplefts = np.dot(homogeneous, matRotate.T)
			bottomrights = bboxes[:,2:]
			tmp = np.ones(len(bottomrights))
			homogeneous = np.hstack([bottomrights, np.expand_dims(tmp, tmp.ndim)])
			final_coord_bottomrights = np.dot(homogeneous, matRotate.T)
			final_anns = np.hstack([final_coord_toplefts, final_coord_bottomrights, np.expand_dims(label, label.ndim)])
		return image, anns


class RandomShiftScale(object):
	def __init__(self,random_ratio,shift,scale):
		self.random_ratio=random_ratio
		self.shift=shift
		self.scale=scale
	def __call__(self, image,bboxes=None):
		if np.random.random() < self.random_ratio:
			s = np.array([image.shape[1], image.shape[0]], dtype=np.float32)
			c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
			sf = self.scale
			cf = self.shift
			c[0] += s[0] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
			c[1] += s[1] * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
			s = s * [np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf),
					 np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)]
			trans_input = get_affine_transform(c, s, 0, [image.shape[1], image.shape[0]])

			image = cv2.warpAffine(image, trans_input,
								 (image.shape[1], image.shape[0]),
								 flags=cv2.INTER_LINEAR)
			bboxes[:,:2] = affine_transform(bboxes[:,:2], trans_input)
			bboxes[:,2:] = affine_transform(bboxes[:,2:], trans_input)
			bboxes[:,[0, 2]] = np.clip(bboxes[:,[0, 2]], 0, image.shape[1] - 1)
			bboxes[:,[1, 3]] = np.clip(bboxes[:,[1, 3]], 0, image.shape[0] - 1)
		return image

class Normalize(object):
	def __init__(self,mean,std):
		self.mean=np.array(mean, dtype=np.float32)
		self.std=np.array(std, dtype=np.float32)
	def __call__(self, image, anns=None, class_name=None):
		image = (image.astype(np.float32) / 255.)
		image = (image - self.mean) / self.std
		image = image.transpose(2, 0, 1)
		return image, anns

class Resize_seg(object):
	def __init__(self, resize_width, resize_height):
		self.resize_width = resize_width
		self.resize_height = resize_height
	def __call__(self, image, anns):
		image = cv2.resize(image,
						(self.resize_width, self.resize_height),
						interpolation=cv2.INTER_LINEAR)
		anns[0] = cv2.resize(anns[0],
						(self.resize_width, self.resize_height),
						interpolation=cv2.INTER_NEAREST)
		return image, anns

class Normalize_seg(object):
	def __init__(self,mean,std):
		self.mean=np.array(mean, dtype=np.float32)
		self.std=np.array(std, dtype=np.float32)
	def __call__(self, image, anns):
		image = (image.astype(np.float32) / 255.)
		image = (image - self.mean) / self.std
		image = image.transpose(2, 0, 1)
		return image, anns

class addGaussianNoise_seg(object):
	def __init__(self):
		pass
	def __call__(self, image, anns):
		globalRandom=np.random.randint(0,101)
		if globalRandom>70:
			G_NoiseNum=int(0.001*image.shape[0]*image.shape[1])
			for i in range(G_NoiseNum):
				temp_x = np.random.randint(0,image.shape[0])
				temp_y = np.random.randint(0,image.shape[1])
				image[temp_x][temp_y] = 255
		return image, anns

class flipHorizon_seg(object):
	def __init__(self):
		pass
	def __call__(self, image, anns):
		img, instance, label = image, anns[0], anns[1]
		globalRandom=np.random.randint(0,101)
		if globalRandom>80:
			img=cv2.flip(img,1)
			instance=cv2.flip(instance,1)
			label = label[::-1]
			# label=np.append(label[:-1][::-1],label[-1]) 
			instance[instance==1]=8
			instance[instance==6]=1
			instance[instance==8]=6
			instance[instance==2]=8
			instance[instance==5]=2
			instance[instance==8]=5
			instance[instance == 3] = 8
			instance[instance == 4] = 3
			instance[instance == 8] = 4
		return img,[instance,label]

class randomCrop_seg(object):
	def __init__(self):
		pass
	def __call__(self, image, anns):
		img, instance = image, anns[0]
		img_h=img.shape[0]
		img_w=img.shape[1]
		globalRandom=np.random.randint(0,101)
		if globalRandom>30:
			rand=np.random.randint(0,101)
			if rand<50:
				seedx=np.random.randint(0,41)/100.0+0.5#0.5-0.9
				seedy=np.random.randint(0,41)/100.0+0.5#0.5-0.9
				dx=np.random.randint(-200,200)
				dy=np.random.randint(-100,200)
				h_new=img_h*seedy
				w_new=img_w*seedx
				img=img[max(0,int(0.5*img_h+dy-0.5*h_new)):min(int(0.5*img_h+dy+0.5*h_new),img_h),\
					max(0,int(0.5*img_w+dx-0.5*w_new)):min(int(0.5*img_w+dx+0.5*w_new),img_w),:]
				instance=instance[max(0,int(0.5*img_h+dy-0.5*h_new)):min(int(0.5*img_h+dy+0.5*h_new),img_h),\
					max(0,int(0.5*img_w+dx-0.5*w_new)):min(int(0.5*img_w+dx+0.5*w_new),img_w)]
				img = cv2.resize(img,(img_w, img_h),interpolation=cv2.INTER_LINEAR)
				instance = cv2.resize(instance,(img_w, img_h),interpolation=cv2.INTER_NEAREST)
			else:
				seedx=np.random.randint(0,41)/100.0+1.1#1.1-1.5
				seedy=np.random.randint(0,41)/100.0+1.1#1.1-1.5
				dx=np.random.randint(-100,200)
				dy=np.random.randint(-100,100)
				h_new=int(img.shape[0]*seedy)
				w_new=int(img.shape[1]*seedx)
				img = cv2.resize(img,(int(w_new),int(h_new)),interpolation=cv2.INTER_LINEAR)
				instance = cv2.resize(instance,(int(w_new),int(h_new)),interpolation=cv2.INTER_NEAREST)
				img=img[max(0,int(0.5*h_new+dy-0.5*img_h)):min(int(0.5*h_new+dy+0.5*img_h),h_new),\
					max(0,int(0.5*w_new+dx-0.5*img_w)):min(int(0.5*w_new+dx+0.5*img_w),w_new),:]
				instance=instance[max(0,int(0.5*h_new+dy-0.5*img_h)):min(int(0.5*h_new+dy+0.5*img_h),h_new),\
					max(0,int(0.5*w_new+dx-0.5*img_w)):min(int(0.5*w_new+dx+0.5*img_w),w_new)]
				img = cv2.resize(img,(img_w, img_h),interpolation=cv2.INTER_LINEAR)
				instance = cv2.resize(instance,(img_w, img_h),interpolation=cv2.INTER_NEAREST)
		return img,[instance, anns[1]]

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