import cv2,torch,torchvision
import torchvision.transforms.functional as F
import numpy as np
from utils.detr_utils import box_xyxy_to_cxcywh, nested_tensor_from_tensor_list

class detrDataset:
	def __init__(self, *args):
		super(detrDataset, self).__init__(*args)
		self.totensor = ToTensor()
		self.normalize = Normalize(self.opt.mean, self.opt.std)
		# type(self).prepare_input = detrDataset.prepare_input

	def get_task_spec_input(self, img, label):
		return img, label

	def preprocess_augment(self, img, anns):
		#to be removed cause model dont care about input size and ratio
		height, width = img.shape[0], img.shape[1]
		if self.opt.keep_res:
			input_h = ((height - 1) | self.opt.pad) + 1  # 获取大于height并能被self.opt.pad整除的最小整数
			input_w = ((width - 1) | self.opt.pad) + 1
		else:
			input_h = ((self.opt.input_h-1)|self.opt.pad) + 1
			input_w = ((self.opt.input_w-1)|self.opt.pad) + 1
		anns = np.array(anns)
		for ann in anns:
			ann[[0, 2]] = ann[[0, 2]] / img.shape[1] * input_w
			ann[[1, 3]] = ann[[1, 3]] / img.shape[0] * input_h
		input_img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_AREA)
		#to be removed cause model dont care about input size and ratio

		#augment
		for item in self.augment_list[:-1]:
			input_img, anns = item(input_img, anns, self.opt.class_name)

		boxes = anns[:, :4]
		boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
		boxes[:, 0::2].clamp_(min=0, max=input_w)
		boxes[:, 1::2].clamp_(min=0, max=input_h)
		
		classes = anns[:, 4]
		classes = torch.tensor(classes, dtype=torch.int64)

		keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
		boxes = boxes[keep]
		classes = classes[keep]

		target = {}
		# target['image_id'] = img_path.split('/')[-1].split('.')[0]
		target['boxes'] = boxes
		target['labels'] = classes
		target['orig_size'] = torch.as_tensor([int(height), int(width)])
		target['size'] = torch.as_tensor([int(input_h), int(input_w)])

		#normalize
		input_img, target = self.totensor(input_img, target)
		input_img, target = self.normalize(input_img, target)

		return input_img, target

	def collate_fn(self, batch):
		batch = list(zip(*batch))
		batch[0] = nested_tensor_from_tensor_list(batch[0])
		return {'img': batch[0], 'annot': batch[1]}

class ToTensor():
	def __call__(self, img, target):
		return F.to_tensor(img), target

class Normalize():
	def __init__(self, mean, std):
		self.mean, self.std = mean, std
	def __call__(self, image, target=None):
		image = F.normalize(image, mean=self.mean, std=self.std)
		if target is None:
			return image, None
		target = target.copy()
		h, w = image.shape[-2:]
		if 'boxes' in target:
			boxes = target['boxes']
			boxes = box_xyxy_to_cxcywh(boxes)
			boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
			target['boxes'] = boxes
		return image, target