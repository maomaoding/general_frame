import time,torch, os
from models import get_model
from models.utils import load_model
from utils.visualizer import Visualizer
import numpy as np

class BaseRecognizor(object):
	def __init__(self, opt):
		if opt.gpus[0] >= 0:
		  opt.device = torch.device('cuda')
		else:
		  opt.device = torch.device('cpu')

		print('Creating model...')
		self.model = get_model(opt)
		self.model = load_model(self.model, opt.model_path)
		# self.model.load_state_dict(torch.load(opt.model_path), strict=False)
		self.model = self.model.to(opt.device)
		self.model.eval()

		self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
		self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
		self.max_per_image = 100
		self.num_classes = opt.num_classes
		self.opt = opt
		self.pause=True

		if opt.export_onnx:
			self.export_onnx()
			os._exit(0)

	def run(self, image):
		load_time, pre_time, net_time, tot_time = 0, 0, 0, 0
		debugger = Visualizer(self.opt)

		start_time = time.time()

		image_tensor = self.prepare_input(image)

		image_tensor = image_tensor.to(self.opt.device)
		pre_process_time = time.time()
		pre_time = pre_process_time - start_time

		raw_output, dets, forward_time = self.process(image_tensor)

		net_time = forward_time - pre_process_time
		end_time = time.time()
		tot_time = end_time - start_time

		self.show_results(debugger, image, dets)

		return {'dets_data': dets, 'dets_image': debugger.imgs['pred_img'], 'tot': tot_time,
				'load': load_time, 'pre': pre_time, 'net': net_time}

	def prepare_input(self, image):
		raise NotImplementedError

	def process(self, image_tensor):
		raise NotImplementedError

	def export_onnx(self):
		raise NotImplementedError

	def show_results(self):
		raise NotImplementedError

	def val_metric(self):
		raise NotImplementedError