import threading
import numpy as np
import os,sys,random,torch
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from torchvision import transforms
import numbers

"""Evaluation Metrics for Semantic Segmentation"""

class SegmentationMetric(object):
	"""Computes pixAcc and mIoU metric scores
	"""

	def __init__(self, nclass):
		super(SegmentationMetric, self).__init__()
		self.nclass = nclass
		self.lock = threading.Lock()
		self.reset()

	def update(self, preds, labels):
		"""Updates the internal evaluation result.
		Parameters
		----------
		labels : 'NumpyArray' or list of `NumpyArray`
			The labels of the data.
		preds : 'NumpyArray' or list of `NumpyArray`
			Predicted values.
		"""
		if isinstance(preds, np.ndarray):
			self.evaluate_worker(preds, labels)
		elif isinstance(preds, (list, tuple)):
			threads = [threading.Thread(target=self.evaluate_worker, args=(pred, label), )
					   for (pred, label) in zip(preds, labels)]
			for thread in threads:
				thread.start()
			for thread in threads:
				thread.join()

	def get(self):
		"""Gets the current evaluation result.
		Returns
		-------
		metrics : tuple of float
			pixAcc and mIoU
		"""
		pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
		IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
		# It has same result with np.nanmean() when all class exist
		mIoU = IoU.mean()
		return pixAcc, mIoU

	def evaluate_worker(self, pred, label):
		correct, labeled = batch_pix_accuracy(pred, label)
		inter, union = batch_intersection_union(pred, label, self.nclass)
		with self.lock:
			self.total_correct += correct
			self.total_label += labeled
			self.total_inter += inter
			self.total_union += union

	def reset(self):
		"""Resets the internal evaluation result to initial state."""
		self.total_inter = 0
		self.total_union = 0
		self.total_correct = 0
		self.total_label = 0


def batch_pix_accuracy(predict, target):
	"""PixAcc"""
	# inputs are numpy array, output 4D, target 3D
	assert predict.shape == target.shape
	predict = predict.astype('int64') + 1
	target = target.astype('int64') + 1

	pixel_labeled = np.sum(target > 0)
	pixel_correct = np.sum((predict == target) * (target > 0))
	assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
	return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
	"""mIoU"""
	# inputs are numpy array, output 4D, target 3D
	assert predict.shape == target.shape
	mini = 1
	maxi = nclass
	nbins = nclass
	predict = predict.astype('int64') + 1
	target = target.astype('int64') + 1

	predict = predict * (target > 0).astype(predict.dtype)
	intersection = predict * (predict == target)
	# areas of intersection and union
	# element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
	area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
	area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
	area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
	area_union = area_pred + area_lab - area_inter
	assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
	return area_inter, area_union

def hist_info(pred, label, num_cls):
	assert pred.shape == label.shape
	k = (label >= 0) & (label < num_cls)
	labeled = np.sum(k)
	correct = np.sum((pred[k] == label[k]))

	return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
																								 num_cls), labeled, correct
class GroupRandomCrop:
	def __init__(self, crop_size):
		self.crop_size = crop_size

	def __call__(self, img_group):
		assert len(img_group) == 3

		img = img_group[0]
		mask = img_group[1]

		ori_height, ori_width = img_group[0].shape[0:2]
		random_scale = random.uniform(0.9, 1.)

		scale_height, scale_width = int(ori_height * random_scale), int(ori_width * random_scale)
		if scale_width > ori_width:
			padh, padw = scale_height - ori_height, scale_width - ori_width
			img = cv2.copyMakeBorder(img_group[0], int(padh/2), padh-int(padh/2), int(padw/2), padw-int(padw/2),
									cv2.BORDER_CONSTANT, value=0)
			mask = cv2.copyMakeBorder(img_group[1], int(padh/2), padh-int(padh/2), int(padw/2), padw-int(padw/2),
									cv2.BORDER_CONSTANT, value=0)
		else:
			x1, y1 = random.randint(0, ori_width - scale_width), random.randint(0, ori_height - scale_height)
			img = img_group[0][y1:y1 + scale_height, x1:x1 + scale_width]
			mask = img_group[1][y1:y1 + scale_height, x1:x1 + scale_width]

		img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR)
		mask = cv2.resize(mask, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_NEAREST)

		return [img, mask, img_group[2]]

class Grouprandomaffine:

	def __call__(self, img_group):
		assert len(img_group) == 3
		ratio = random.uniform(-0.2, 0.2)

		pts1 = np.float32([[0, 0], [0, img_group[0].shape[0]-1], [img_group[0].shape[1]-1, img_group[0].shape[0]-1]])
		pts2 = np.float32([[ratio*img_group[0].shape[1], 0], [0, img_group[0].shape[0]-1],
							[img_group[0].shape[1]-1, img_group[0].shape[0]-1]])
		M = cv2.getAffineTransform(pts1, pts2)

		img = cv2.warpAffine(img_group[0], M, (img_group[0].shape[1], img_group[0].shape[0]), flags=cv2.INTER_LINEAR)
		mask = cv2.warpAffine(img_group[1], M, (img_group[1].shape[1], img_group[1].shape[0]), flags=cv2.INTER_NEAREST)

		return [img, mask, img_group[2]]

class GroupRandomBlur:

	def __call__(self, img_group):
		assert len(img_group) == 3
		if random.random() < 0.5:
			img_group[0] = cv2.GaussianBlur(img_group[0], (5,5), random.uniform(1e-6, 0.6))

		return img_group

class GroupRandomHorizontalFlip:

	def __call__(self, img_group):
		assert len(img_group) == 3
		
		if random.random() < 0.5:
			img_group[0] = np.ascontiguousarray(np.fliplr(img_group[0]))
			img_group[1] = np.ascontiguousarray(np.fliplr(img_group[1]))

			'''for culane dataset'''
			# tmp = 5-img_group[1]
			# img_group[1] = tmp * (tmp >= 1) * (tmp <= 4)
			'''for our own dataset nclasses=8'''
			#extract classes: 1,2,3,4,5,6 and stopline: 7
			flipline = img_group[1] * (img_group[1] >= 1) * (img_group[1] <= 6)
			stopline = img_group[1] * (img_group[1] == 7)
			#start flip label
			flipline = 7 - flipline
			flipline = flipline * (flipline >= 1) * (flipline <= 6)
			#final merge
			img_group[1] = flipline + stopline

			img_group[2] = img_group[2][::-1]

		return img_group

class GroupNormalize:
	def __init__(self, mean, std):
		self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean, std),
			])

	def __call__(self, img_group):
		assert len(img_group) == 3

		img_group[0] = self.transform(img_group[0])
		img_group[1] = torch.from_numpy(img_group[1]).long()
		img_group[2] = torch.from_numpy(np.array(img_group[2]))

		return img_group

class GroupRandomCropRatio(object):
	def __init__(self, size):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size

	def __call__(self, img_group):
		h, w = img_group[0].shape[0:2]
		tw, th = self.size

		out_images = list()
		h1 = random.randint(0, max(0, h - th))
		w1 = random.randint(0, max(0, w - tw))
		h2 = min(h1 + th, h)
		w2 = min(w1 + tw, w)

		for img in img_group[:2]:
			assert (img.shape[0] == h and img.shape[1] == w)
			out_images.append(img[h1:h2, w1:w2, ...])
		out_images.append(img_group[2])
		return out_images
class GroupRandomScale(object):
	def __init__(self, size=(0.5, 1.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)):
		self.size = size
		self.interpolation = interpolation

	def __call__(self, img_group):
		assert (len(self.interpolation) == len(img_group[:2]))
		scale = random.uniform(self.size[0], self.size[1])
		out_images = list()
		for img, interpolation in zip(img_group, self.interpolation):
			out_images.append(cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation))
			if len(img.shape) > len(out_images[-1].shape):
				out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
		out_images.append(img_group[2])
		return out_images
class GroupRandomRotation(object):
	def __init__(self, degree=(-10, 10), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=None):
		self.degree = degree
		self.interpolation = interpolation
		self.padding = padding

	def __call__(self, img_group):
		assert (len(self.interpolation) == len(img_group[:2]))
		v = random.random()
		if v < 0.5:
			degree = random.uniform(self.degree[0], self.degree[1])
			h, w = img_group[0].shape[0:2]
			center = (w / 2, h / 2)
			map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
			out_images = list()
			for img, interpolation, padding in zip(img_group, self.interpolation, self.padding):
				out_images.append(cv2.warpAffine(img, map_matrix, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=padding))
				if len(img.shape) > len(out_images[-1].shape):
					out_images[-1] = out_images[-1][..., np.newaxis]  # single channel image
			out_images.append(img_group[2])
			return out_images
		else:
			return img_group