import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat
import torch.nn.functional as F

def _reg_loss(regr, gt_regr, mask):
	''' L1 regression loss
	  Arguments:
		regr (batch x max_objects x dim)
		gt_regr (batch x max_objects x dim)
		mask (batch x max_objects)
	'''
	num = mask.float().sum()
	mask = mask.unsqueeze(2).expand_as(gt_regr).float()

	regr = regr * mask
	gt_regr = gt_regr * mask

	regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
	regr_loss = regr_loss / (num + 1e-4)
	return regr_loss

def _neg_loss(pred, gt):
	''' Modified focal loss. Exactly the same as CornerNet.
		Runs faster and costs a little bit more memory
	  Arguments:
		pred (batch x c x h x w)
		gt_regr (batch x c x h x w)
	'''
	pos_inds = gt.eq(1).float()
	neg_inds = gt.lt(1).float()

	neg_weights = torch.pow(1 - gt, 4)

	loss = 0

	pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
	neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

	num_pos = pos_inds.float().sum()
	pos_loss = pos_loss.sum()
	neg_loss = neg_loss.sum()

	if num_pos == 0:
		loss = loss - neg_loss
	else:
		loss = loss - (pos_loss + neg_loss) / num_pos
	return loss


class FocalLoss(nn.Module):
	'''nn.Module warpper for focal loss'''

	def __init__(self):
		super(FocalLoss, self).__init__()
		self.neg_loss = _neg_loss

	def forward(self, out, target):
		return self.neg_loss(out, target)

class RegL1Loss(nn.Module):
	def __init__(self):
		super(RegL1Loss, self).__init__()

	def forward(self, output, mask, ind, target):
		pred = _tranpose_and_gather_feat(output, ind)
		mask = mask.unsqueeze(2).expand_as(pred).float()
		# loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
		loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
		loss = loss / (mask.sum() + 1e-4)
		return loss

class RegLoss(nn.Module):
	'''Regression loss for an output tensor
	  Arguments:
		output (batch x dim x h x w)
		mask (batch x max_objects)
		ind (batch x max_objects)
		target (batch x max_objects x dim)
	'''

	def __init__(self):
		super(RegLoss, self).__init__()

	def forward(self, output, mask, ind, target):
		pred = _tranpose_and_gather_feat(output, ind)
		loss = _reg_loss(pred, target, mask)
		return loss

class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        #mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss