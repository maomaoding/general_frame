import os, cv2
import torch.utils.data as data
from ..augment.transforms import *
import re

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def natural_key(string_):
	"""See http://www.codinghorror.com/blog/archives/001018.html"""
	return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def load_class_map(filename, root=''):
	class_map_path = filename
	if not os.path.exists(class_map_path):
		class_map_path = os.path.join(root, filename)
		assert os.path.exists(class_map_path), 'Cannot locate specified class map file ({})'.format(filename)
	class_map_ext = os.path.splitext(filename)[-1].lower()
	if class_map_ext == '.txt':
		with open(class_map_path) as f:
			class_to_idx = {v.strip(): k for k, v in enumerate(f)}
	else:
		assert False, 'Unsupported class map extension'
	return class_to_idx

def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
	labels = []
	filenames = []
	for root, subdirs, files in os.walk(folder, topdown=False):
		rel_path = os.path.relpath(root, folder) if (root != folder) else ''
		label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
		for f in files:
			base, ext = os.path.splitext(f)
			if ext.lower() in types:
				filenames.append(os.path.join(root, f))
				labels.append(label)
	if class_to_idx is None:
		# building class index
		unique_labels = set(labels)
		sorted_labels = list(sorted(unique_labels, key=natural_key))
		class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
	images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
	if sort:
		images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
	return images_and_targets, class_to_idx

class Furnitures(data.Dataset):
	def __init__(self, opt, split='train'):
		super(Furnitures, self).__init__()

		self.data_dir = os.path.join(opt.data_dir, opt.dataset)
		self.root = os.path.join(self.data_dir, split)
		self.class_to_idx = None
		if opt.class_map:
			self.class_to_idx = load_class_map(opt.class_map, self.root)
		images, self.class_to_idx = find_images_and_targets(self.root, class_to_idx=self.class_to_idx)
		if len(images) == 0:
				raise RuntimeError(f'Found 0 images in subfolders of {root}. '
													 f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
		self.samples = images
		self.imgs = self.samples # torchvision ImageFolder compat

		if split == 'train':
			self.augment_list = [RandomGaussBlursub(0.3), RandomNoisesub(0.3), Normalizesub(opt.mean, opt.std)]
		else:
			self.augment_list = [Normalizesub(opt.mean, opt.std)]
		self.opt = opt
		self.split = split

	def __len__(self):
		return len(self.samples)

	def get_img_path(self, index):
		return self.samples[index][0]

	def get_ann(self, index):
		return self.samples[index][1]

	def prepare_input(self, img_path, anns):
		img = cv2.imread(img_path)
		img = cv2.resize(img, (self.opt.input_w, self.opt.input_h), interpolation=cv2.INTER_AREA)

		#for data augmentation
		for item in self.augment_list:
			img = item(img)

		if anns is None:
			anns = torch.zeros(1).long()
		return img, anns

# data augment transform
class RandomGaussBlursub(RandomGaussBlur):
	def __init__(self, random_ratio):
		super().__init__(random_ratio)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)[0]

class RandomNoisesub(RandomNoise):
	def __init__(self, random_ratio):
		super().__init__(random_ratio)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)[0]

class Normalizesub(Normalize):
	def __init__(self, mean,std):
		super().__init__(mean,std)
	def __call__(self, image, anns=None, class_name=None):
		return super().__call__(image, anns)[0]