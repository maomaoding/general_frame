from .base_detector import BaseDetector
import cv2,torch,time,os
import numpy as np

class Unetppdet(BaseDetector):
	def prepare_input(self, image):
		resized_image = cv2.resize(image,
						(self.opt.resize_width, self.opt.resize_height),
						interpolation=cv2.INTER_LINEAR)
		inp_image = ((resized_image / 255. - self.mean) / self.std).astype(np.float32)
		inp_image = inp_image.transpose(2, 0, 1).reshape(1, 3, self.opt.resize_height, self.opt.resize_width)
		inp_image = torch.from_numpy(inp_image)
		return inp_image

	def process(self, image):
		output = self.model(image)
		forward_time = time.time()
		return output, output['spatial'][-1].detach().cpu().numpy(), forward_time

	def show_results(self, debugger, image, dets, output):
		height, width = image.shape[0:2]
		predictmap = np.argmax(dets[0, ...], axis=0)
		bmap = (predictmap==1)*0+(predictmap==2)*255+(predictmap==3)*0+(predictmap==4)*0\
				+(predictmap==5)*255+(predictmap==6)*97+(predictmap==7)*125
		gmap = (predictmap==1)*97+(predictmap==2)*0+(predictmap==3)*255+(predictmap==4)*0\
		 		+(predictmap==5)*255+(predictmap==6)*0+(predictmap==7)*125
		rmap = (predictmap==1)*255+(predictmap==2)*0+(predictmap==3)*0+(predictmap==4)*255\
				+(predictmap == 5)*0+(predictmap == 6)*255+(predictmap == 7)*0
		colormap = np.stack((bmap, gmap, rmap), axis=2).astype(np.uint8)
		colormap = cv2.resize(colormap, (width, height), interpolation=cv2.INTER_LINEAR)
		dst = cv2.addWeighted(image.astype(np.uint8), 0.7, colormap, 0.3, 0)

		# labeldict = {0:'lack',1:'yellowsolid',2:'whitesolid',3:'yellowdash',4:'whitedash'}
		# predict_label = output['label'][0].detach().cpu().numpy()
		# predict_label = np.argmax(predict_label, axis=0)
		# str_label = 'predict: '
		# for tt in predict_label:
		# 	str_label += labeldict[tt] + ' '

		# cv2.putText(dst, str_label, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
		debugger.add_img(dst, img_id='out_pred')
		debugger.show_all_imgs(pause=True)

	def export_onnx(self):
		dummy_input = torch.randn(1, 3, self.opt.resize_height, self.opt.resize_width, device='cuda')
		output=["spatial","label"]
		self.model.eval()
		print("start exporting onnx")
		torch.onnx.export(self.model, dummy_input, self.opt.model_onnx_path, verbose=True,
						input_names=["data"],output_names=output)
		print("onnx export complete")