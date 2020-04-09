import os, cv2, json
import numpy as np
import torch.utils.data as data

class TT100k(data.Dataset):
	def get_ann(self, index):
		label_id = self.label_list[index]
		label_json = os.path.join(self.data_dir, "data", "annotations.json")

		with open(label_json, 'r') as f:
			json_f = json.load(f)
			if str(label_id) in json_f['imgs']:
				label_file = json_f['imgs'][str(label_id)]['objects']
			else:
				return []
			labels=[]
			for obj in label_file:
				cls_id = 0
				bbox = obj['bbox']
				bb = (float(bbox['xmin']), float(bbox['ymin']),
					float(bbox['xmax']), float(bbox['ymax']))

				label = [ bb[0], bb[1], bb[2], bb[3],cls_id]
				labels.append(label)
			labels = np.array(labels)

		return labels

	def get_img_path(self, index):
		return self.img_path[index]

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
		anns=anns.tolist()
		input_img=cv2.resize(input_img,(input_w,input_h))

		return	input_img, anns

	def __init__(self, opt, split='train'):
		super(TT100k, self).__init__(opt, split)
		self.data_dir = os.path.join(opt.data_dir, opt.dataset)
		self.img_dir = []
		# self.img_dir = os.path.join(self.data_dir, "data/{}".format(split))
		self.img_dir.append(os.path.join(self.data_dir, "data/{}".format(split)))
		if split == 'train':
			self.img_dir.append(os.path.join(self.data_dir, "data/{}".format('nosign_5')))
		self.label_dir = os.path.join(self.data_dir, "data")

		self.opt = opt
		self.split = split

		print('==> initializing TT100K {} data.'.format(split))
		self.img_path=[]
		self.label_list=[]
		for i in range(len(self.img_dir)):
			for file in os.listdir(self.img_dir[i]):
				self.img_path.append(os.path.join(self.img_dir[i],file))
				self.label_list.append(file.split('.')[0])

		self.num_samples = len(self.img_path)
		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples