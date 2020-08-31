import numpy as np
import cv2
class Visualizer(object):
	def __init__(self,opt):
		self.color_list = np.array(
			[
				1.000, 1.000, 1.000,
				0.850, 0.325, 0.098,
				0.929, 0.694, 0.125,
				0.494, 0.184, 0.556,
				0.466, 0.674, 0.188,
				0.301, 0.745, 0.933,
				0.635, 0.078, 0.184,
				0.300, 0.300, 0.300,
				0.600, 0.600, 0.600,
				1.000, 0.000, 0.000,
				1.000, 0.500, 0.000,
				0.749, 0.749, 0.000,
				0.000, 1.000, 0.000,
				0.000, 0.000, 1.000,
				0.667, 0.000, 1.000,
				0.333, 0.333, 0.000,
				0.333, 0.667, 0.000,
				0.333, 1.000, 0.000,
				0.667, 0.333, 0.000,
				0.667, 0.667, 0.000,
				0.667, 1.000, 0.000,
				1.000, 0.333, 0.000,
				1.000, 0.667, 0.000,
				1.000, 1.000, 0.000,
				0.000, 0.333, 0.500,
				0.000, 0.667, 0.500,
				0.000, 1.000, 0.500,
				0.333, 0.000, 0.500,
				0.333, 0.333, 0.500,
				0.333, 0.667, 0.500,
				0.333, 1.000, 0.500,
				0.667, 0.000, 0.500,
				0.667, 0.333, 0.500,
				0.667, 0.667, 0.500,
				0.667, 1.000, 0.500,
				1.000, 0.000, 0.500,
				1.000, 0.333, 0.500,
				1.000, 0.667, 0.500,
				1.000, 1.000, 0.500,
				0.000, 0.333, 1.000,
				0.000, 0.667, 1.000,
				0.000, 1.000, 1.000,
				0.333, 0.000, 1.000,
				0.333, 0.333, 1.000,
				0.333, 0.667, 1.000,
				0.333, 1.000, 1.000,
				0.667, 0.000, 1.000,
				0.667, 0.333, 1.000,
				0.667, 0.667, 1.000,
				0.667, 1.000, 1.000,
				1.000, 0.000, 1.000,
				1.000, 0.333, 1.000,
				1.000, 0.667, 1.000,
				0.167, 0.000, 0.000,
				0.333, 0.000, 0.000,
				0.500, 0.000, 0.000,
				0.667, 0.000, 0.000,
				0.833, 0.000, 0.000,
				1.000, 0.000, 0.000,
				0.000, 0.167, 0.000,
				0.000, 0.333, 0.000,
				0.000, 0.500, 0.000,
				0.000, 0.667, 0.000,
				0.000, 0.833, 0.000,
				0.000, 1.000, 0.000,
				0.000, 0.000, 0.167,
				0.000, 0.000, 0.333,
				0.000, 0.000, 0.500,
				0.000, 0.000, 0.667,
				0.000, 0.000, 0.833,
				0.000, 0.000, 1.000,
				0.000, 0.000, 0.000,
				0.143, 0.143, 0.143,
				0.286, 0.286, 0.286,
				0.429, 0.429, 0.429,
				0.571, 0.571, 0.571,
				0.714, 0.714, 0.714,
				0.857, 0.857, 0.857,
				0.000, 0.447, 0.741,
				0.50, 0.5, 0
			]
		).astype(np.float32)
		self.color_list = self.color_list.reshape((-1, 3)) * 255

		colors = [(self.color_list[_]).astype(np.uint8) \
				  for _ in range(len(self.color_list))]
		self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
		self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
		self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
		if hasattr(opt, 'class_name'):
			self.names=opt.class_name
		if hasattr(opt, 'down_ratio'):
			self.down_ratio = opt.down_ratio
		self.imgs = {}

	def add_img(self, img, img_id='default', revert_color=False):
		if revert_color:
			img = 255 - img
		self.imgs[img_id] = img.copy()

	def add_mask(self, mask, bg, imgId='default', trans=0.8):
		self.imgs[imgId] = (mask.reshape(
			mask.shape[0], mask.shape[1], 1) * 255 * trans + \
							bg * (1 - trans)).astype(np.uint8)

	def show_img(self, pause=False, imgId='default'):
		cv2.imshow('{}'.format(imgId), self.imgs[imgId])
		if pause:
			cv2.waitKey()

	def add_rect(self, rect1, rect2, c, conf=1, img_id='default'):
		cv2.rectangle(
			self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
		if conf < 1:
			cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
			cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
			cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
			cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

	def gen_colormap(self, img, output_res=None):
		img = img.copy()
		c, h, w = img.shape[0], img.shape[1], img.shape[2]
		if output_res is None:
			output_res = (h * self.down_ratio, w * self.down_ratio)
		img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
		colors = np.array(
			self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)

		colors = 255 - colors
		color_map = (img * colors).max(axis=2).astype(np.uint8)
		color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
		return color_map

	def add_blend_img(self, back, fore, img_id='blend', trans=0.7):

		fore = 255 - fore
		if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
			fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
		if len(fore.shape) == 2:
			fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
		self.imgs[img_id] = (back * (1. - trans) + fore * trans)
		self.imgs[img_id][self.imgs[img_id] > 255] = 255
		self.imgs[img_id][self.imgs[img_id] < 0] = 0
		self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

	def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'):
		bbox = np.array(bbox, dtype=np.int32)
		# cat = (int(cat) + 1) % 80
		cat = int(cat)
		# print('cat', cat, self.names[cat])
		c = self.colors[cat][0][0].tolist()

		c = (255 - np.array(c)).tolist()
		txt = '{}{:.1f}'.format(self.names[cat], conf)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
		cv2.rectangle(
			self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
		if show_txt:
			cv2.rectangle(self.imgs[img_id],
						  (bbox[0], bbox[1] - cat_size[1] - 2),
						  (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
			cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2),
						font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

	def add_points(self, points, img_id='default'):
		num_classes = len(points)
		# assert num_classes == len(self.colors)
		for i in range(num_classes):
			for j in range(len(points[i])):
				c = self.colors[i, 0, 0]
				cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
											   points[i][j][1] * self.down_ratio),
						   5, (255, 255, 255), -1)
				cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
											   points[i][j][1] * self.down_ratio),
						   3, (int(c[0]), int(c[1]), int(c[2])), -1)

	def show_all_imgs(self, pause=False, time=0):
		for i, v in self.imgs.items():
			cv2.namedWindow('{}'.format(i), cv2.WINDOW_GUI_NORMAL)
			cv2.imshow('{}'.format(i), v)
		if cv2.waitKey(0 if pause else 1) == 27:
			import sys
			sys.exit(0)