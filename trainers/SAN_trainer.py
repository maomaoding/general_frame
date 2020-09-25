from base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import *

class SoftTargetCrossEntropy(nn.Module):
	def __init__(self):
		super(SoftTargetCrossEntropy, self).__init__()

	def forward(self, x, batch):
		target = batch['annot']
		loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
		loss_stats = {'loss': loss.mean()}
		return loss.mean(), loss_stats

class LabelSmoothingCrossEntropy(nn.Module):
	"""
	NLL loss with label smoothing.
	"""
	def __init__(self, smoothing=0.1):
		"""
		Constructor for the LabelSmoothing module.
		:param smoothing: label smoothing factor
		"""
		super(LabelSmoothingCrossEntropy, self).__init__()
		assert smoothing < 1.0
		self.smoothing = smoothing
		self.confidence = 1. - smoothing

	def forward(self, x, batch):
		target = batch['annot']
		logprobs = F.log_softmax(x, dim=-1)
		nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = self.confidence * nll_loss + self.smoothing * smooth_loss
		loss_stats = {'loss': loss.mean()}
		return loss.mean(), loss_stats

class CrossEntropyLoss(nn.Module):
	def __init__(self):
		super(CrossEntropyLoss, self).__init__()
		self.loss_fn = torch.nn.CrossEntropyLoss()

	def forward(self, x, batch):
		loss = self.loss_fn(x, batch['annot'])
		loss_stats = {'loss': loss}
		return loss, loss_stats

@register_trainer
class SAN_trainer(BaseTrainer):
	def __init__(self, opt):
		super(SAN_trainer, self).__init__(opt)

	def gen_optimizer(self):
		return torch.optim.Adam(self.model.parameters(), self.opt.lr)

	def _get_losses(self, opt):
		loss_states = ['loss',]
		mixup_active = opt.mixup > 0 or opt.cutmix > 0. or opt.cutmix_minmax is not None
		if mixup_active:
			# smoothing is handled with mixup target transform
			loss = SoftTargetCrossEntropy().cuda()
		elif opt.smoothing:
			loss = LabelSmoothingCrossEntropy(smoothing=opt.smoothing).cuda()
		else:
			loss = CrossEntropyLoss().cuda()
		return loss_states, loss

	def model_with_loss(self, batch):
		outputs = self.model(batch['img'])
		loss, loss_stats = self.loss(outputs, batch)
		return outputs, loss, loss_stats