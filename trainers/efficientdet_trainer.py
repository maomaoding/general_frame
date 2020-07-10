import torch, cv2
import numpy as np
from .base_trainer import BaseTrainer
from models.networks.efficientdet import BBoxTransform, ClipBoxes, postprocess

def calc_iou(a, b):
	# a(anchor) [boxes, (y1, x1, y2, x2)]
	# b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

	area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
	iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
	ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
	iw = torch.clamp(iw, min=0)
	ih = torch.clamp(ih, min=0)
	ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
	ua = torch.clamp(ua, min=1e-8)
	intersection = iw * ih
	IoU = intersection / ua

	return IoU

class FocalLoss(torch.nn.Module):
	def __init__(self):
		super(FocalLoss, self).__init__()

	def forward(self, outputs, batch):
		_, regressions, classifications, anchors = outputs
		annotations = batch['annot']
		alpha = 0.25
		gamma = 2.0
		batch_size = classifications.shape[0]
		classification_losses = []
		regression_losses = []

		anchor = anchors[0, :, :]
		dtype = anchors.dtype

		anchor_widths = anchor[:, 3] - anchor[:, 1]
		anchor_heights = anchor[:, 2] - anchor[:, 0]
		anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
		anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

		for j in range(batch_size):
			classification = classifications[j, :, :]
			regression = regressions[j, :, :]

			bbox_annotation = annotations[j]
			bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

			classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

			if bbox_annotation.shape[0] == 0:
				if torch.cuda.is_available():
					alpha_factor = torch.ones_like(classification) * alpha
					alpha_factor = alpha_factor.cuda()
					alpha_factor = 1. - alpha_factor
					focal_weight = classification
					focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

					bce = -(torch.log(1.0 - classification))
					cls_loss = focal_weight * bce

					regression_losses.append(torch.tensor(0).to(dtype).cuda())
					classification_losses.append(cls_loss.sum())
				else:
					alpha_factor = torch.ones_like(classification) * alpha
					alpha_factor = 1. - alpha_factor
					focal_weight = classification
					focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

					bce = -(torch.log(1.0 - classification))
					cls_loss = focal_weight * bce

					regression_losses.append(torch.tensor(0).to(dtype))
					classification_losses.append(cls_loss.sum())
				continue

			IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

			IoU_max, IoU_argmax = torch.max(IoU, dim=1)

			# compute the loss for classification
			targets = torch.ones_like(classification) * -1
			if torch.cuda.is_available():
				targets = targets.cuda()

			targets[torch.lt(IoU_max, 0.4), :] = 0

			positive_indices = torch.ge(IoU_max, 0.5)

			num_positive_anchors = positive_indices.sum()

			assigned_annotations = bbox_annotation[IoU_argmax, :]

			targets[positive_indices, :] = 0
			targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

			alpha_factor = torch.ones_like(targets) * alpha
			if torch.cuda.is_available():
				alpha_factor = alpha_factor.cuda()

			alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
			focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
			focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

			bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

			cls_loss = focal_weight * bce

			zeros = torch.zeros_like(cls_loss)
			if torch.cuda.is_available():
				zeros = zeros.cuda()
			cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

			classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

			if positive_indices.sum() > 0:
				assigned_annotations = assigned_annotations[positive_indices, :]

				anchor_widths_pi = anchor_widths[positive_indices]
				anchor_heights_pi = anchor_heights[positive_indices]
				anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
				anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

				gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
				gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
				gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
				gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

				# efficientdet style
				gt_widths = torch.clamp(gt_widths, min=1)
				gt_heights = torch.clamp(gt_heights, min=1)

				targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
				targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
				targets_dw = torch.log(gt_widths / anchor_widths_pi)
				targets_dh = torch.log(gt_heights / anchor_heights_pi)

				targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
				targets = targets.t()

				regression_diff = torch.abs(targets - regression[positive_indices, :])

				regression_loss = torch.where(
					torch.le(regression_diff, 1.0 / 9.0),
					0.5 * 9.0 * torch.pow(regression_diff, 2),
					regression_diff - 0.5 / 9.0
				)
				regression_losses.append(regression_loss.mean())
			else:
				if torch.cuda.is_available():
					regression_losses.append(torch.tensor(0).to(dtype).cuda())
				else:
					regression_losses.append(torch.tensor(0).to(dtype))
		reg_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True).mean()
		cls_loss = torch.stack(classification_losses).mean(dim=0, keepdim=True).mean()
		loss = reg_loss + cls_loss
		loss_stats = {'loss': loss, 'reg_loss': reg_loss, 'cls_loss': cls_loss}
		return loss, loss_stats

class EfficientDetTrainer(BaseTrainer):
	def __init__(self, opt):
		super(EfficientDetTrainer, self).__init__(opt)

	def _get_losses(self, opt):
		loss_states = ['loss', 'reg_loss', 'cls_loss']
		loss = FocalLoss()
		return loss_states, loss

	def gen_optimizer(self):
		return torch.optim.AdamW(self.model.parameters(), self.opt.lr)

	def visual(self, vis, batch, output):
		features, regression, classification, anchors = output
		imgs, annots = batch['img'], batch['annot']

		img = imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
		img = np.clip(((img * self.opt.std + self.opt.mean) * 255.), 0, 255).astype(np.uint8)

		regressBoxes = BBoxTransform()
		clipBoxes = ClipBoxes()
		threshold = 0.2
		iou_threshold = 0.2

		with torch.no_grad():
			out = postprocess(imgs[0:1,...],
							anchors, regression, classification,
							regressBoxes, clipBoxes,
							threshold, iou_threshold)

		# vis.add_img(img, img_id='pred')
		# for j in range(len(out[0]['rois'])):
		# 	bboxes = out[0]['rois'][j].astype(np.int)
		# 	obj = out[0]['class_ids'][j]
		# 	score = float(out[0]['scores'][j])
		# 	vis.add_coco_bbox(bboxes, obj, score, img_id='pred')

		# vis.add_img(img, img_id='gt')
		# for j in range(len(annots[0])):
		# 	if annots[0, j, -1] != -1:
		# 		bboxes = annots[0,j,:4]
		# 		obj = annots[0,j,4]
		# 		score = 1
		# 		vis.add_coco_bbox(bboxes.cpu(), obj.cpu(), score, img_id='gt')

		# vis.show_all_imgs(pause=False)