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
			label = [float(xmin), float(ymin), float(xmax), float(ymax), int(classid)]
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
			input_img, anns = item(input_img, anns, self.opt.class_name)

		return input_img, anns

	def __init__(self, opt, split='train'):
		super(Junctions, self).__init__()
		self.data_dir = os.path.join(opt.data_dir, opt.dataset)
		self.train_root = os.path.join(self.data_dir, split)

		self.opt = opt
		self.split = split
		if split == 'train':# RandomCropsub(0.5,0.1),RandomFlipsub(0.5),
			self.augment_list = [RandomRotate90(0.5),
								RandomGaussBlursub(0.3), RandomContrastBrightsub(0.3), RandomNoisesub(0.3),
								Normalizesub(opt.mean, opt.std)]
		else:
			self.augment_list = [Normalizesub(opt.mean, opt.std)]

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

class RandomFlipsub(RandomFlip):
	def __init__(self, random_ratio):
		super().__init__(random_ratio)
	def __call__(self, image, anns=None, class_name=None):
		if np.random.random()<self.random_ratio:
			horizontal_or_vertical = np.random.random()
			height, width = image.shape[:2]
			if horizontal_or_vertical < 0.5: #horizontal
				image = image[:, ::-1, :]
				anns[:, [0, 2]] = width - anns[:, [2, 0]] - 1
				for i in range(len(anns)):
					old_classname = class_name[int(anns[i, 4])]
					if old_classname.find('right') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('right', 'left'))
					elif old_classname.find('left') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('left', 'right'))
			else:							 #vertical
				image = image[::-1, :, :]
				anns[:, [1, 3]] = height - anns[:, [3, 1]] - 1
				for i in range(len(anns)):
					old_classname = class_name[int(anns[i, 4])]
					if old_classname.find('top') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('top', 'bottom'))
					elif old_classname.find('bottom') != -1:
						anns[i, 4] = class_name.index(old_classname.replace('bottom', 'top'))
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
				old_classname = class_name[int(label[i])]
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
			new_toplefts = np.dot(homogeneous, matRotate.T)
			bottomrights = bboxes[:,2:]
			tmp = np.ones(len(bottomrights))
			homogeneous = np.hstack([bottomrights, np.expand_dims(tmp, tmp.ndim)])
			new_bottomrights = np.dot(homogeneous, matRotate.T)
			final_toplefts = (new_toplefts + new_bottomrights) * 0.5 - abs(new_toplefts - new_bottomrights) * 0.5
			final_bottomrights = (new_toplefts + new_bottomrights) * 0.5 + abs(new_toplefts - new_bottomrights) * 0.5
			anns = np.hstack([final_toplefts, final_bottomrights, np.expand_dims(label, label.ndim)])
		return image, anns

class RandomCropsub(RandomCrop):
	def __init__(self, random_ratio, shift):
		super().__init__(random_ratio, shift)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)

class RandomGaussBlursub(RandomGaussBlur):
	def __init__(self, random_ratio):
		super().__init__(random_ratio)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)

class RandomContrastBrightsub(RandomContrastBright):
	def __init__(self, random_ratio):
		super().__init__(random_ratio)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)

class RandomNoisesub(RandomNoise):
	def __init__(self, random_ratio):
		super().__init__(random_ratio)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)

class Normalizesub(Normalize):
	def __init__(self, mean,std):
		super().__init__(mean,std)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)