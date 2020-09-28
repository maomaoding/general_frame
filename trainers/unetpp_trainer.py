from .base_trainer import BaseTrainer
import torch,cv2,os
import numpy as np

class Unetpploss(torch.nn.Module):
	def __init__(self, opt):
		super(Unetpploss, self).__init__()
		self.criterion_spatial = torch.nn.CrossEntropyLoss(weight=torch.tensor(opt.lane_weight) \
															if isinstance(opt.lane_weight, list) \
															else opt.lane_weight)
		if opt.num_labels == 4:
			self.criterion_label = torch.nn.CrossEntropyLoss()
		elif opt.num_labels == 7:
			self.criterion_label = torch.nn.BCELoss()
		self.opt = opt

	def forward(self, outputs, batch):
		opt = self.opt
		spatial_loss = 0
		for predictmap in outputs['spatial']:
			spatial_loss += self.criterion_spatial(predictmap, batch['instance'])
		label_loss = self.criterion_label(outputs['label'], batch['label'])
		loss = spatial_loss * opt.loss_weight[0] + label_loss * opt.loss_weight[1]
		loss_states = {'loss': loss, 'spatial_loss': spatial_loss, 'label_loss': label_loss}
		return loss, loss_states


class UnetppTrainer(BaseTrainer):
	def __init__(self, opt):
		super(UnetppTrainer, self).__init__(opt)

	def _set_optimizer_param(self, net):
		params = [{"params": net.parameters(), "lr": self.opt.lr},]
		return params

	def _get_losses(self, opt):
		loss_states = ['loss', 'spatial_loss', 'label_loss']
		loss = Unetpploss(opt)
		return loss_states, loss

	def visual(self, vis, batch, output, avg_loss_stats=None):
		predictmap = output['spatial'][-1].detach()
		for i in range(predictmap.shape[0]):
			ith_predictmap = predictmap[i, ...].cpu().numpy()
			ith_predictmap = np.argmax(ith_predictmap, axis=0)
			bmap = (ith_predictmap==1)*0+(ith_predictmap==2)*255+(ith_predictmap==3)*0+(ith_predictmap==4)*0\
					+(ith_predictmap==5)*255+(ith_predictmap==6)*97+(ith_predictmap==7)*125
			gmap = (ith_predictmap==1)*97+(ith_predictmap==2)*0+(ith_predictmap==3)*255+(ith_predictmap==4)*0\
			 		+(ith_predictmap==5)*255+(ith_predictmap==6)*0+(ith_predictmap==7)*125
			rmap = (ith_predictmap==1)*255+(ith_predictmap==2)*0+(ith_predictmap==3)*0+(ith_predictmap==4)*255\
					+(ith_predictmap == 5)*0+(ith_predictmap == 6)*255+(ith_predictmap == 7)*0
			colormap = np.stack((bmap, gmap, rmap), axis=2).astype(np.uint8)
			frame = batch['input'][i].detach().cpu().numpy().transpose(1,2,0) * np.array(self.opt.std, dtype=np.float32)\
					+ np.array(self.opt.mean, dtype=np.float32)
			frame = frame * 255
			dst = cv2.addWeighted(frame.astype(np.uint8), 0.7, colormap, 0.3, 0)

			if self.opt.num_labels == 4:
				labeldict = {0:'lack',1:'yellowsolid',2:'whitesolid',3:'yellowdash',4:'whitedash'}
				predict_label = output['label'][i].detach().cpu().numpy()
				predict_label = np.argmax(predict_label, axis=0)
				str_label = 'predict: '
				for tt in predict_label:
					str_label += labeldict[tt] + ' '

				target_label = batch['label'][i].detach().cpu().numpy()
				str_targetlabel = 'target: '
				for tt in target_label:
					str_targetlabel += labeldict[tt] + ' '

				cv2.putText(dst, str_label, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
				cv2.putText(dst, str_targetlabel, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

			vis.img("predict img batch_{}".format(i), torch.from_numpy(dst.transpose(2,0,1)))
			if avg_loss_stats != None:
				for l in avg_loss_stats:
					vis.plot(l, avg_loss_stats[l].avg)
			# cv2.imshow('{}'.format(i), dst)
			# if cv2.waitKey(1) == 27:
			# 	import sys
			# 	sys.exit(0)

	def model_with_loss(self, batch):
		outputs = self.model(batch['img'])
		loss, loss_stats = self.loss(outputs, batch)
		return outputs, loss, loss_stats