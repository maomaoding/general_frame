import torch.nn as nn
import torch
import numpy as np

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
	topk_ys = (topk_inds / width).int().float()
	topk_xs = (topk_inds % width).int().float()

	topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
	topk_clses = (topk_ind / K).int()
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

def _voc_ap(rec, prec, use_07_metric=False):
	"""Compute VOC AP given precision and recall. If use_07_metric is true, uses
	the VOC 07 11-point method (default:False).
	"""
	if use_07_metric:
		# 11 point metric
		ap = 0.
		for t in np.arange(0., 1.1, 0.1):
			if np.sum(rec >= t) == 0:
				p = 0
			else:
				p = np.max(prec[rec >= t])
			ap = ap + p / 11.
	else:
		# correct AP calculation
		# first append sentinel values at the end
		mrec = np.concatenate(([0.], rec, [1.]))
		mpre = np.concatenate(([0.], prec, [0.]))

		# compute the precision envelope
		for i in range(mpre.size - 1, 0, -1):
			mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

		# to calculate area under PR curve, look for points
		# where X axis (recall) changes value
		i = np.where(mrec[1:] != mrec[:-1])[0]

		# and sum (\Delta recall) * prec
		ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	return ap