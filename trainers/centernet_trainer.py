from .base_trainer import BaseTrainer
from utils.losses import FocalLoss, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from utils.utils import _sigmoid, _nms, _topk, _tranpose_and_gather_feat
from utils.visualizer import Visualizer
import torch
import numpy as np

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
	batch, cat, height, width = heat.size()

	heat = _nms(heat)

	scores, inds, clses, ys, xs = _topk(heat, K=K)
	if reg is not None:
		reg = _tranpose_and_gather_feat(reg, inds)
		reg = reg.view(batch, K, 2)
		xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
		ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
	else:
		xs = xs.view(batch, K, 1) + 0.5
		ys = ys.view(batch, K, 1) + 0.5
	wh = _tranpose_and_gather_feat(wh, inds)
	if cat_spec_wh:
		wh = wh.view(batch, K, cat, 2)
		clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
		wh = wh.gather(2, clses_ind).view(batch, K, 2)
	else:
		wh = wh.view(batch, K, 2)
	clses = clses.view(batch, K, 1).float()
	scores = scores.view(batch, K, 1)
	bboxes = torch.cat([xs - wh[..., 0:1] / 2,
						ys - wh[..., 1:2] / 2,
						xs + wh[..., 0:1] / 2,
						ys + wh[..., 1:2] / 2], dim=2)
	detections = torch.cat([bboxes, scores, clses], dim=2)

	return detections

class CenterNetLoss(torch.nn.Module):
	def __init__(self, opt):
		super(CenterNetLoss, self).__init__()
		self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
		self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
			RegLoss() if opt.reg_loss == 'sl1' else None
		self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
			NormRegL1Loss() if opt.norm_wh else \
			RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
		self.opt = opt

	def forward(self, outputs, batch):
		opt = self.opt
		hm_loss, wh_loss, off_loss = 0, 0, 0
		if not opt.mse_loss:
			outputs['hm'] = _sigmoid(outputs['hm'])
		hm_loss += self.crit(outputs['hm'], batch['hm'])
		if opt.wh_weight > 0:
			if opt.dense_wh:
				mask_weight = batch['dense_wh_mask'].sum() + 1e-4
				wh_loss += (
								   self.crit_wh(outputs['wh'] * batch['dense_wh_mask'],
												batch['dense_wh'] * batch['dense_wh_mask']) /
								   mask_weight)
			elif opt.cat_spec_wh:
				wh_loss += self.crit_wh(
					outputs['wh'], batch['cat_spec_mask'],
					batch['ind'], batch['cat_spec_wh'])
			else:
				wh_loss += self.crit_wh(
					outputs['wh'], batch['reg_mask'],
					batch['ind'], batch['wh'])

		if opt.reg_offset and opt.off_weight > 0:
			off_loss += self.crit_reg(outputs['reg'], batch['reg_mask'],
									  batch['ind'], batch['reg'])

		loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
				opt.off_weight * off_loss
		loss_stats = {'loss': loss, 'hm_loss': hm_loss,
					  'wh_loss': wh_loss, 'off_loss': off_loss}
		return loss, loss_stats


class CenterNetTrainer(BaseTrainer):
	def __init__(self, opt):
		super(CenterNetTrainer, self).__init__(opt)

	def _set_optimizer_param(self, net):
		params = [{"params": net.parameters(), "lr": self.opt.lr},]
		return params

	def _get_losses(self, opt):
		loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
		loss = CenterNetLoss(opt)
		return loss_states, loss

	def visual(self, vis, batch, output, avg_loss_stats=None):
		opt = self.opt
		reg = output['reg'] if opt.reg_offset else None
		dets = ctdet_decode(
			output['hm'], output['wh'], reg=reg,
			cat_spec_wh=opt.cat_spec_wh, K=opt.max_objs)
		dets = dets.detach().cpu().numpy().reshape(dets.shape[0], -1, dets.shape[2])
		dets[:, :, :4] *= opt.down_ratio
		dets_gt = batch['gt'].numpy().reshape(dets.shape[0], -1, dets.shape[2])
		dets_gt[:, :, :4] *= opt.down_ratio
		debugger = Visualizer(opt)
		img = batch['input'][0].detach().cpu().numpy().transpose(1, 2, 0)
		img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
		pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
		gt = debugger.gen_colormap(batch['hm'][0].detach().cpu().numpy())
		# debugger.add_blend_img(img, pred, 'pred_hm')
		# debugger.add_blend_img(img, gt, 'gt_hm')
		debugger.add_img(img, img_id='out_pred')
		for k in range(len(dets[0])):
			if dets[0, k, 4] > self.opt.vis_thresh:
				debugger.add_coco_bbox(dets[0, k, :4], dets[0, k, -1],
										dets[0, k, 4], img_id='out_pred')

		debugger.add_img(img, img_id='out_gt')
		for k in range(len(dets_gt[0])):
			if dets_gt[0, k, 4] > 0:
				debugger.add_coco_bbox(dets_gt[0, k, :4], dets_gt[0, k, -1],
									   dets_gt[0, k, 4], img_id='out_gt')
		if opt.visual:
			debugger.show_all_imgs(pause=False)