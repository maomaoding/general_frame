import torch.utils.data as data
import os,cv2

class CowaLane(data.Dataset):
	def get_img_path(self,index):
		return self.img_paths[index]

	def get_ann(self, index):
		return self.instance_paths[index], self.labels[index]

	def prepare_input(self, img_path, anns):
		assert type(anns[0]) == str and type(anns[1]) == list
		img = cv2.imread(img_path)
		instance = cv2.imread(anns[0], 0)
		return img, [instance, anns[1]]

	def __init__(self, opt, split='train'):
		super(CowaLane, self).__init__(opt, split)
		self.data_dir = os.path.join(opt.data_dir, opt.dataset)
		self.data_txt = os.path.join(self.data_dir, "{}.txt".format(split))

		self.opt = opt

		print("==> initializing CowaLane {} data".format(split))
		self.img_paths = []
		self.instance_paths = []
		self.labels = []
		with open(self.data_txt, 'r') as f:
			for line in f:
				img_path,instance_path,line1,line2,line3,line4,line5,line6,line7 = line.rstrip().split(' ')
				self.img_paths.append(os.path.join(self.data_dir, img_path))
				self.instance_paths.append(os.path.join(self.data_dir, instance_path))
				# self.labels.append([float(line2),float(line3),float(line4),float(line5)])
				self.labels.append([float(float(line1)>0), float(float(line2)>0), float(float(line3)>0), float(float(line4)>0),
									float(float(line5)>0), float(float(line6)>0), float(float(line7)>0)])
		self.num_samples = len(self.img_paths)
		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples

class Culane(data.Dataset):
	def get_img_path(self, index):
		return self.img_paths[index]

	def get_ann(self, index):
		return self.instance_paths[index], self.labels[index]

	def prepare_input(self, img_path, anns):
		assert type(anns[0]) == str and type(anns[1]) == list
		img = cv2.imread(img_path)
		instance = cv2.imread(anns[0], 0)
		return img, [instance, anns[1]]

	def __init__(self, opt, split='train'):
		super(Culane, self).__init__(opt, split)
		self.data_dir = os.path.join(opt.data_dir, opt.dataset)
		self.data_txt = os.path.join(self.data_dir, "{}_gt.txt".format(split))

		print("==> initializing Culane {} data".format(split))
		self.img_paths = []
		self.instance_paths = []
		self.labels = []
		with open(self.data_txt, 'r') as f:
			for line in f:
				img_path, instance_path, ll, el, er, rr = line.rstrip().split(' ')
				self.img_paths.append(os.path.join(self.data_dir, img_path[1:]))
				self.instance_paths.append(os.path.join(self.data_dir, instance_path[1:]))
				self.labels.append([float(ll),float(el),float(er),float(rr)])
		self.num_samples = len(self.img_paths)
		print('Loaded {} {} samples'.format(split, self.num_samples))

	def __len__(self):
		return self.num_samples