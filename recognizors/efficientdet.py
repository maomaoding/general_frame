from .base_recognizor import BaseRecognizor
import os,cv2,torch,time,json,shutil
import numpy as np
from utils.utils import voc_ap
from collections import defaultdict
from models.networks.efficientdet import BBoxTransform, ClipBoxes, postprocess

class Efficientdet(BaseRecognizor):
	def prepare_input(self, image):
		height, width = image.shape[:2]
		if self.opt.keep_res:
			self.opt.input_h = ((height - 1) | self.opt.pad) + 1
			self.opt.input_w = ((width - 1) | self.opt.pad) + 1
		else:
			self.opt.input_h = ((self.opt.input_h-1)|self.opt.pad) + 1
			self.opt.input_w = ((self.opt.input_w-1)|self.opt.pad) + 1
		resized_image = cv2.resize(image,(self.opt.input_w,self.opt.input_h), interpolation=cv2.INTER_AREA)
		inp_image = ((resized_image / 255. - self.mean) / self.std).astype(np.float32)
		inp_image = inp_image.transpose(2, 0, 1).reshape(1, 3, self.opt.input_h, self.opt.input_w)
		inp_image = torch.from_numpy(inp_image)
		return inp_image

	def process(self, image):
		output = self.model(image)
		_, regression, classification, anchors = output
		forward_time = time.time()

		regressBoxes = BBoxTransform()
		clipBoxes = ClipBoxes()
		threshold = 0.2
		iou_threshold = 0.2

		with torch.no_grad():
			out = postprocess(image,
							anchors, regression, classification,
							regressBoxes, clipBoxes,
							threshold, iou_threshold)
		return output, out, forward_time

	def show_results(self, debugger, image, out):
		debugger.add_img(image, img_id='pred_img')
		height, width = image.shape[:2]
		for j in range(len(out[0]['rois'])):
			bboxes = out[0]['rois'][j].astype(np.int)
			bboxes[[0, 2]] = bboxes[[0, 2]] / self.opt.input_w * width
			bboxes[[1, 3]] = bboxes[[1, 3]] / self.opt.input_h * height
			obj = out[0]['class_ids'][j]
			score = float(out[0]['scores'][j])
			debugger.add_coco_bbox(bboxes, obj, score, img_id='pred_img')
		if self.opt.show_results:
			debugger.show_all_imgs(pause=self.pause)

	def export_onnx(self):
		input_h = ((self.opt.input_h - 1) | self.opt.pad) + 1
		input_w = ((self.opt.input_w - 1) | self.opt.pad) + 1
		dummy_input = torch.randn(1, 3, input_h, input_w, device='cuda')
		self.model.eval()
		torch.onnx.export(self.model, dummy_input, self.opt.model_onnx_path, verbose=True,
						input_names=["data"])

	def val_metric(self, ovthresh=0.5):
		data_dir = os.path.join(self.opt.data_dir, self.opt.dataset)
		val_root = os.path.join(data_dir, 'val')

		TEMP_FILES_PATH = ".temp_files"
		if not os.path.exists(TEMP_FILES_PATH):
			os.makedirs(TEMP_FILES_PATH)

		gt_counter_per_class = {}
		dets_result = [[] for _ in range(self.opt.num_classes)]

		for i, path in enumerate(open(os.path.join(data_dir, 'val.txt'), 'r')):
			img, label = path.rstrip().split(' ')
			imgpath = os.path.join(val_root, img)
			labelpath = os.path.join(val_root, label)
			file_id = os.path.basename(img).split('.')[0]

			img = cv2.imread(imgpath)
			height, width = img.shape[0], img.shape[1]
			ret = self.run(img)

			#gt specific to this img
			# create ground-truth dictionary
			bounding_boxes = []
			with open(labelpath, 'r') as f:
				for line in f:
					xmin, ymin, xmax, ymax, cls = line.rstrip().split(' ')
					bounding_boxes.append({"class_name":cls, "bbox":" ".join([xmin,ymin,xmax,ymax]),
										"used":False})
					if cls in gt_counter_per_class:
						gt_counter_per_class[cls] += 1
					else:
						gt_counter_per_class[cls] = 1
			# dump bounding_boxes into a ".json" file
			new_temp_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
			with open(new_temp_file, 'w') as outfile:
				json.dump(bounding_boxes, outfile)

			#dets to this img
			ratio_w = width / self.opt.input_w
			ratio_h = height / self.opt.input_h
			objs = ret['dets_data'][0]
			for j in range(len(objs['rois'])):
				bbox = objs['rois'][j]
				bbox[[0,2]] = bbox[[0,2]] * ratio_w
				bbox[[1,3]] = bbox[[1,3]] * ratio_h
				bbox = bbox.round().astype(np.int)
				cls = objs['class_ids'][j]
				score = float(objs['scores'][j])
				dets_result[int(cls)].append({"confidence":str(score), "file_id": file_id,
										"bbox":" ".join(bbox.astype(str))})

		for key, value in gt_counter_per_class.items():
			dets_result[int(key)].sort(key=lambda x:float(x['confidence']), reverse=True)
			with open(TEMP_FILES_PATH + "/" + key + "_dr.json", 'w') as outfile:
				json.dump(dets_result[int(key)], outfile)

		"""
		Calculate the AP for each class
		"""
		sum_AP = 0.0
		count_true_positives = {}
		for class_name in list(gt_counter_per_class.keys()):
			count_true_positives[class_name] = 0
			"""
			Load detection-results of that class
			"""
			dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
			dr_data = json.load(open(dr_file))
			"""
			Assign detection-results to ground-truth objects
			"""
			nd = len(dr_data)
			tp = [0] * nd
			fp = [0] * nd
			for idx, detection in enumerate(dr_data):
				file_id = detection['file_id']
				# assign detection-results to ground truth object if any
				# open ground-truth with that file_id
				gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
				ground_truth_data = json.load(open(gt_file))
				ovmax = -1
				gt_match = -1
				#load detected object bounding-box
				bb = [float(x) for x in detection['bbox'].split()]
				for obj in ground_truth_data:
					#look for a class_name match
					if obj['class_name'] == class_name:
						bbgt = [float(x) for x in obj['bbox'].split()]
						bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
						iw = bi[2] - bi[0] + 1
						ih = bi[3] - bi[1] + 1
						if iw > 0 and ih > 0:
							# compute overlap (IoU) = area of intersection / area of union
							ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
									+ 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
							ov = iw * ih / ua
							if ov > ovmax:
								ovmax = ov
								gt_match = obj

				#set minimum overlap
				min_overlap = ovthresh
				if ovmax >= min_overlap:
					if not bool(gt_match['used']):
						tp[idx] = 1
						gt_match['used'] = True
						count_true_positives[class_name] += 1
						#update the ".json" file
						with open(gt_file, 'w') as f:
							f.write(json.dumps(ground_truth_data))
					else:
						fp[idx] = 1
				else:
					fp[idx] = 1
			#compute precision/recall
			cumsum = 0
			for idx, val in enumerate(fp):
				fp[idx] += cumsum
				cumsum += val
			cumsum = 0
			for idx, val in enumerate(tp):
				tp[idx] += cumsum
				cumsum += val
			rec = tp[:]
			for idx, val in enumerate(tp):
				rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
			prec = tp[:]
			for idx, val in enumerate(tp):
				prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

			ap, mrec, mprec = voc_ap(rec[:], prec[:])
			sum_AP += ap

		mAP = sum_AP / len(list(gt_counter_per_class.keys()))
		shutil.rmtree(TEMP_FILES_PATH)
		return mAP