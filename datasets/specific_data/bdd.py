import torch.utils.data as data
import numpy as np
import xml.etree.ElementTree as ET
import os,cv2

class BDD(data.Dataset):
	def get_ann(self, index):
		label_path = self.label_path[index]
		tree = ET.parse(label_path)
		root = tree.getroot()
		size = root.find('size')
		labels=[]
		for obj in root.iter('object'):
			cls = obj.find('name').text
			if cls not in self.opt.class_name:
				continue
			cls_id = self.opt.class_name.index(cls)
			xmlbox = obj.find('bndbox')
			bb = (
				float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
				float(xmlbox.find('ymax').text))

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

	def __init__(self, opt, split="train"):
		super(BDD, self).__init__(opt, split)
		self.data_dir = os.path.join(opt.data_dir, opt.dataset)
		self.img_dir = os.path.join(self.data_dir, "images")
		self.label_dir = os.path.join(self.data_dir, "labels")

		self.annot_path = os.path.join(self.data_dir,'{}.txt').format(split)

		self.opt = opt
		self.split = split

		print('==> initializing bdd100k {} data.'.format(split))
		self.img_path=[]
		self.label_path=[]
		for path in open(self.annot_path, 'r'):
			self.img_path.append(os.path.join( self.img_dir,"100k/{}/".format(split),path.rstrip()+".jpg"))
			self.label_path.append(os.path.join(self.label_dir, "{}/".format(split), path.rstrip()+ ".xml"))
		self.num_samples = len(self.img_path)
		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples