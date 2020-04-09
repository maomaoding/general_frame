from trainers.centernet_trainer import ctdet_decode
from .base_detector import BaseDetector
import cv2,torch,time,os,json
import numpy as np
from utils.utils import _voc_ap

class CenterNetdet(BaseDetector):
	def prepare_input(self, image):
		height, width = image.shape[0:2]
		if self.opt.keep_res:
			self.opt.input_h = ((height - 1) | self.opt.pad) + 1
			self.opt.input_w = ((width - 1) | self.opt.pad) + 1
		else:
			self.opt.input_h = ((self.opt.input_h-1)|self.opt.pad) + 1
			self.opt.input_w = ((self.opt.input_w-1)|self.opt.pad) + 1
		resized_image = cv2.resize(image,(self.opt.input_w,self.opt.input_h))
		inp_image = ((resized_image / 255. - self.mean) / self.std).astype(np.float32)
		inp_image = inp_image.transpose(2, 0, 1).reshape(1, 3, self.opt.input_h, self.opt.input_w)
		inp_image = torch.from_numpy(inp_image)
		return inp_image

	def process(self, image):
		output = self.model(image)
		hm = output['hm'].sigmoid_()
		wh = output['wh']
		reg = output['reg'] if self.opt.reg_offset else None
		forward_time = time.time()
		dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.max_objs)
		return output, dets, forward_time

	def show_results(self, debugger, image, dets, output, scale=1):
		detection = dets.detach().cpu().numpy().copy()
		pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
		detection[:, :, [0, 2]] = detection[:, :, [0, 2]] * self.opt.down_ratio
		detection[:, :, [1, 3]] = detection[:, :, [1, 3]] * self.opt.down_ratio
		image=cv2.resize(image,pred.shape[:2])
		debugger.add_blend_img(image, pred, 'pred_hm_{:.1f}'.format(scale))
		debugger.add_img(image, img_id='out_pred_{:.1f}'.format(scale))
		for k in range(len(dets[0])):
			if detection[0, k, 4] > self.opt.vis_thresh:
				debugger.add_coco_bbox(detection[0, k, :4], detection[0, k, -1],
									   detection[0, k, 4],
									   img_id='out_pred_{:.1f}'.format(scale))
		debugger.show_all_imgs(pause=self.pause)

	def export_onnx(self):
		input_h = ((self.opt.input_h - 1) | self.opt.pad) + 1
		input_w = ((self.opt.input_w - 1) | self.opt.pad) + 1
		dummy_input = torch.randn(1, 3, input_h, input_w, device='cuda')
		output=["wh","reg","hm"]
		self.model.eval()
		torch.onnx.export(self.model, dummy_input, self.opt.model_onnx_path, verbose=True,
						input_names=["data"],output_names=output)

	def val_metric(self):
		self.pause = False
		val_filepath = self.opt.val_filepath
		data_dir = os.path.join(self.opt.data_dir, self.opt.dataset)
		label_json = os.path.join(data_dir, "data", "annotations.json")

		npos = 0
		confidence_tpfp_pair = []

		with open(label_json, 'r') as f:
			json_f = json.load(f)
			for file in os.listdir(val_filepath):
				img = cv2.imread(os.path.join(val_filepath, file))
				ret = self.run(img)

				#gt specific to this img
				label_file = json_f['imgs'][file.split('.')[0]]['objects']
				BBGT = [[obj['bbox']['xmin'], obj['bbox']['ymin'],
						obj['bbox']['xmax'], obj['bbox']['ymax']] for obj in label_file]
				BBGT = np.array(BBGT)
				height, width = img.shape[0], img.shape[1]
				h_ratio, w_ratio = height / self.opt.input_h, width / self.opt.input_w
				BBGT[:, [0,2]] = BBGT[:, [0,2]] / w_ratio
				BBGT[:, [1,3]] = BBGT[:, [1,3]] / h_ratio

				det = [False] * len(BBGT)
				npos += len(BBGT)

				bboxes = ret['results'].detach().cpu().numpy()
				bboxes = np.array(sorted(bboxes[0], key = lambda x:x[4], reverse=True))
				for bb in bboxes:
					if bb[4] > self.opt.vis_thresh:
						tp, fp = 0, 0
						ovmax = -np.inf
						if BBGT.size > 0:
							#compute overlaps
							# intersection
							ixmin = np.maximum(BBGT[:, 0], bb[0]*4)
							iymin = np.maximum(BBGT[:, 1], bb[1]*4)
							ixmax = np.minimum(BBGT[:, 2], bb[2]*4)
							iymax = np.minimum(BBGT[:, 3], bb[3]*4)
							iw = np.maximum(ixmax - ixmin + 1., 0.)
							ih = np.maximum(iymax - iymin + 1., 0.)
							inters = iw * ih

							# union
							uni = ((bb[2]*4 - bb[0]*4 + 1.) * (bb[3]*4 - bb[1]*4 + 1.) +
									(BBGT[:, 2] - BBGT[:, 0] + 1.) *
									(BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
							overlaps = inters / uni
							ovmax = np.max(overlaps)
							jmax = np.argmax(overlaps)
						if ovmax > 0.5:
							if not det[jmax]:
								tp = 1.
								det[jmax] = 1
							else:
								fp = 1.
						else:
							fp = 1.
						confidence_tpfp_pair.append([bb[4], tp, fp])
		confidence_tpfp_pair = np.array(sorted(confidence_tpfp_pair, key = lambda x:x[0], reverse=True))
		fp = np.cumsum(confidence_tpfp_pair[:, 2])
		tp = np.cumsum(confidence_tpfp_pair[:, 1])
		rec = tp / float(npos)
		prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
		ap = _voc_ap(rec, prec, use_07_metric=False)
		return rec, prec, ap