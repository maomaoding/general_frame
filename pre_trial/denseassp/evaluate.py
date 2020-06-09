import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os,torch,cv2
from denseaspp import *
import numpy as np
from torchvision import transforms
import threading


def batch_pix_accuracy(predict, target):
	assert predict.shape == target.shape
	predict = predict.astype('int64') + 1
	target = target.astype('int64') + 1

	pixel_labeled = np.sum(target > 0)
	pixel_correct = np.sum((predict == target) * (target > 0))
	assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
	return pixel_correct, pixel_labeled

def batch_intersection_union(predict, target, nclass):
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

class SegmentationMetric(object):
	def __init__(self, nclass):
		self.nclass = nclass
		self.lock = threading.Lock()
		self.reset()

	def update(self, preds, labels):
		if isinstance(preds, np.ndarray):
			self.evaluate_worker(preds, labels)
		elif isinstance(preds, (list, tuple)):
			threads = [threading.Thread(target=self.evaluate_worker, args=(pred, label),)
						for (pred, label) in zip(preds, labels)]
			for thread in threads:
				thread.start()
			for thread in threads:
				thread.join()

	def get(self):
		pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
		IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)

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
		self.total_inter = 0
		self.total_union = 0
		self.total_correct = 0
		self.total_label = 0

input_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
])

cuda0 = torch.device('cuda:1')

pretrained_weight = '/home/ubuntu/dyh/denseassp/snapshot/denseaspp_curbstonedata.pth'

model = get_denseaspp(pretrained_weight=pretrained_weight)
model.eval()
model = model.to(cuda0)

'''for a bunch of test img'''
pred = []
label = []

val_txt = '/home/ubuntu/dyh/curbstonedata/dyh/val.txt'

with open(val_txt, 'r') as f:
	for line in f.readlines():
		line = line.rstrip().split()
		img = cv2.imread(line[0])
		img = img[:,:,::-1]
		img = cv2.resize(img, (480, 480))

		img = input_transform(img)
		img = torch.unsqueeze(img, 0)
		img = img.to(cuda0)

		out = model(img)
		out = torch.argmax(out, 1)
		out = torch.squeeze(out)
		out = out.cpu().numpy()

		pred.append(out)
		target = cv2.imread(line[1], 0)
		target = cv2.resize(target, (480,480))
		label.append(target)

metric = SegmentationMetric(nclass=2)

metric.update(pred, label)

pixAcc, mIoU = metric.get()
print('validation pixAcc: %.3f%%, mIoU: %.3f%%' % (pixAcc * 100, mIoU * 100))