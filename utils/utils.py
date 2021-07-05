import torch.nn as nn
import torch
import numpy as np
import torch.distributed as dist

class AverageMeter:
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		if self.count > 0:
			self.avg = self.sum / self.count

def _nms(heat, kernel=3):
	pad = (kernel - 1) // 2

	hmax = nn.functional.max_pool2d(
		heat, (kernel, kernel), stride=1, padding=pad)
	keep = (hmax == heat).float()
	return heat * keep

def _topk(scores, K=40):
	batch, cat, height, width = scores.size()

	topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

	topk_inds = topk_inds % (height * width)
	topk_ys = (topk_inds // width).int().float()
	topk_xs = (topk_inds % width).int().float()

	topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
	topk_clses = (topk_ind // K).int()
	topk_inds = _gather_feat(
		topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
	topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
	topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

	return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _sigmoid(x):
	y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
	return y

def _gather_feat(feat, ind, mask=None):
	dim  = feat.size(2)
	ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
	feat = feat.gather(1, ind)
	if mask is not None:
		mask = mask.unsqueeze(2).expand_as(feat)
		feat = feat[mask]
		feat = feat.view(-1, dim)
	return feat

def _tranpose_and_gather_feat(feat, ind):
	feat = feat.permute(0, 2, 3, 1).contiguous()
	feat = feat.view(feat.size(0), -1, feat.size(3))
	feat = _gather_feat(feat, ind)
	return feat

"""
 Calculate the AP given the recall and precision array
	1st) We compute a version of the measured precision/recall curve with
		 precision monotonically decreasing
	2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
	"""
	--- Official matlab code VOC2012---
	mrec=[0 ; rec ; 1];
	mpre=[0 ; prec ; 0];
	for i=numel(mpre)-1:-1:1
			mpre(i)=max(mpre(i),mpre(i+1));
	end
	i=find(mrec(2:end)~=mrec(1:end-1))+1;
	ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
	"""
	rec.insert(0, 0.0) # insert 0.0 at begining of list
	rec.append(1.0) # insert 1.0 at end of list
	mrec = rec[:]
	prec.insert(0, 0.0) # insert 0.0 at begining of list
	prec.append(0.0) # insert 0.0 at end of list
	mpre = prec[:]
	"""
	 This part makes the precision monotonically decreasing
		(goes from the end to the beginning)
		matlab: for i=numel(mpre)-1:-1:1
					mpre(i)=max(mpre(i),mpre(i+1));
	"""
	# matlab indexes start in 1 but python in 0, so I have to do:
	#     range(start=(len(mpre) - 2), end=0, step=-1)
	# also the python function range excludes the end, resulting in:
	#     range(start=(len(mpre) - 2), end=-1, step=-1)
	for i in range(len(mpre)-2, -1, -1):
		mpre[i] = max(mpre[i], mpre[i+1])
	"""
	 This part creates a list of indexes where the recall changes
		matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
	"""
	i_list = []
	for i in range(1, len(mrec)):
		if mrec[i] != mrec[i-1]:
			i_list.append(i) # if it was matlab would be i + 1
	"""
	 The Average Precision (AP) is the area under the curve
		(numerical integration)
		matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
	"""
	ap = 0.0
	for i in i_list:
		ap += ((mrec[i]-mrec[i-1])*mpre[i])
	return ap, mrec, mpre

def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True

def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()

def is_main_process():
	return get_rank() == 0

def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()