
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2,os
from datasets.augment.image import get_affine_transform,affine_transform
def _get_border( border, size):
	i = 1
	while size - border // i <= border // i:
		i *= 2
	return border // i

#RandomCrop和RandomShiftScale只可用一个，RandomNoise只能在RandomCrop（RandomShiftScale）和RandomGasussBlur之后使用
#输入的bbox为一个二维list，第一维是目标个数，第二维是[x1,y1,x2,y2]
class DataAugment(object):
	def __init__(self, aug_list):
		self.aug_list = aug_list
	def __call__(self, image, bboxes=None):
		if bboxes is not None and len(bboxes)>0:
			for aug in self.aug_list:
				bboxes_np=np.array(bboxes)[:,:4]
				extra_label=np.array(bboxes)[:,4:]
				image=aug(image,bboxes_np)
				bboxes_np=np.hstack([bboxes_np,extra_label])
				bboxes=bboxes_np.tolist()
			return image,bboxes
		elif len(bboxes)==0:
			for aug in self.aug_list:
				image=aug(image)
			return image, bboxes
		else:
			for aug in self.aug_list:
				image=aug(image)
			return image

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
	def __call__(self, image, bboxes=None):
		if np.random.random()<self.random_ratio:
			alpha=np.random.random()+0.3
			beta=int((np.random.random()-0.5)*255*0.3)
			blank = np.zeros(image.shape, image.dtype)
			image = cv2.addWeighted(image, alpha, blank, 1 - alpha, beta)
			image=np.clip(image,0,255)
		return image

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

class RandomGasussBlur(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, bboxes=None):
		if np.random.random()<self.random_ratio:
			kernel_size = (np.random.randint(1,3)*2+1, np.random.randint(1,3)*2+1)
			sigma = 0.8
			image = cv2.GaussianBlur(image, kernel_size, sigma)
		return image

class RandomNoise(object):
	def __init__(self,random_ratio):
		self.random_ratio=random_ratio
	def __call__(self, image, bboxes=None):
		if np.random.random()<self.random_ratio:
			noise_num = int(0.001 * image.shape[0] * image.shape[1])
			for i in range(noise_num):
				temp_x = np.random.randint(0, image.shape[0])
				temp_y = np.random.randint(0, image.shape[1])
				image[temp_x][temp_y] = 255
		return image

class RandomCrop(object):
	def __init__(self,random_ratio, shift):
		self.shift = shift
		self.random_ratio=random_ratio
	def __call__(self, image,bboxes=[]):
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
			if bboxes == []:
				return image
			bboxes[:,:2] = affine_transform(bboxes[:,:2], trans_input)
			bboxes[:,2:] = affine_transform(bboxes[:,2:], trans_input)
			bboxes[:,[0, 2]] = np.clip(bboxes[:,[0, 2]], 0, image.shape[1] - 1)
			bboxes[:,[1, 3]] = np.clip(bboxes[:,[1, 3]], 0, image.shape[0] - 1)
		return image

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
	def __call__(self, image, bboxes=None):
		image = (image.astype(np.float32) / 255.)
		image = (image - self.mean) / self.std
		image = image.transpose(2, 0, 1)
		return image

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
	img=cv2.imread("/datastore/data/dataset/coco/train2014/COCO_train2014_000000236177.jpg")
	bbox=np.array([[198.04,  20.8 , 478.92 ,569.61,1],[401.16, 381.19 ,465.42 ,470.68,2]])

	aug=DataAugment([RandomCrop(1, 0.1),])
	for i in bbox:
		cv2.rectangle(img,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,0,0))
	cv2.imshow("ori", img)
	img = cv2.imread("/datastore/data/dataset/coco/train2014/COCO_train2014_000000236177.jpg")

	img_aug,bbox=aug(img,bbox)
	for i in bbox:
		cv2.rectangle(img_aug,(int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(255,255,0))
	cv2.imshow("aug",img_aug)
	cv2.waitKey()