import math
from .base import BaseTask
import numpy as np

def gaussian_radius(det_size, min_overlap=0.7):
	height, width = det_size

	# r1:            r2:              r3:
	#    ____       ____________      _____w______
	#  r|__  |     |            |    |            |
	# | |\\| |     |    w       |    |            |
	#h| |\\| |     |   |\\\\|   |    |   |\\\\|   |
	# | |\\|_|     |   |\\\\|h  |    |   |\\\\|   |h
	# |____|       | r |\\\\|   |    | r |\\\\|   |
	#    w         |            |    |            |
	#              |____________|    |____________|
	#
	#

	a1 = 1
	b1 = (height + width)
	c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
	sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
	r1 = (b1 - sq1) / (2 * a1)

	a2 = 4
	b2 = 2 * (height + width)
	c2 = (1 - min_overlap) * width * height
	sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
	r2 = (b2 - sq2) / (2 * a2)

	a3 = 4 * min_overlap
	b3 = -2 * min_overlap * (height + width)
	c3 = (min_overlap - 1) * width * height
	sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
	r3 = (b3 + sq3) / (2 * a3)
	return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
	m, n = [(ss - 1.) / 2. for ss in shape]
	y, x = np.ogrid[-m:m + 1, -n:n + 1]

	h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
	h[h < np.finfo(h.dtype).eps * h.max()] = 0
	return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
	diameter = 2 * radius + 1
	gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

	x, y = int(center[0]), int(center[1])

	height, width = heatmap.shape[0:2]

	left, right = min(x, radius), min(width - x, radius + 1)
	top, bottom = min(y, radius), min(height - y, radius + 1)

	masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
	masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
	if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
		np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
	return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
	diameter = 2 * radius + 1
	gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
	value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
	dim = value.shape[0]
	reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
	if is_offset and dim == 2:
		delta = np.arange(diameter * 2 + 1) - radius
		reg[0] = reg[0] - delta.reshape(1, -1)
		reg[1] = reg[1] - delta.reshape(-1, 1)

	x, y = int(center[0]), int(center[1])

	height, width = heatmap.shape[0:2]

	left, right = min(x, radius), min(width - x, radius + 1)
	top, bottom = min(y, radius), min(height - y, radius + 1)

	masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
	masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
	masked_gaussian = gaussian[radius - top:radius + bottom,
					  radius - left:radius + right]
	masked_reg = reg[:, radius - top:radius + bottom,
				 radius - left:radius + right]
	if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
		idx = (masked_gaussian - masked_heatmap<1e-8).reshape(
			1, masked_gaussian.shape[0], masked_gaussian.shape[1])
		masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
	regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
	return regmap


def draw_msra_gaussian(heatmap, center, sigma):
	tmp_size = sigma * 3
	mu_x = int(center[0] + 0.5)
	mu_y = int(center[1] + 0.5)
	w, h = heatmap.shape[0], heatmap.shape[1]
	ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
	br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
	if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
		return heatmap
	size = 2 * tmp_size + 1
	x = np.arange(0, size, 1, np.float32)
	y = x[:, np.newaxis]
	x0 = y0 = size // 2
	g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
	g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
	g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
	img_x = max(0, ul[0]), min(br[0], h)
	img_y = max(0, ul[1]), min(br[1], w)
	heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
		heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
		g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
	return heatmap

class CTDetDataset(BaseTask):
	def get_data(self,img,anns):
		num_objs = min(len(anns), self.opt.max_objs)
		height, width = img.shape[1], img.shape[2]#img :c h w

		output_h = height // self.opt.down_ratio
		output_w = width // self.opt.down_ratio
		num_classes = self.opt.num_classes

		hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
		wh = np.zeros((self.opt.max_objs, 2), dtype=np.float32)
		dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
		reg = np.zeros((self.opt.max_objs, 2), dtype=np.float32)
		ind = np.zeros((self.opt.max_objs), dtype=np.int64)
		reg_mask = np.zeros((self.opt.max_objs), dtype=np.uint8)
		cat_spec_wh = np.zeros((self.opt.max_objs, num_classes * 2), dtype=np.float32)
		cat_spec_mask = np.zeros((self.opt.max_objs, num_classes * 2), dtype=np.uint8)
		gt_det=np.zeros((self.opt.max_objs, 6), dtype=np.float32)
		draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
			draw_umich_gaussian

		for k in range(num_objs):
			ann = np.array(anns[k])
			bbox = ann[:4]
			cls_id = int(ann[4])
			bbox[[0, 2]] = np.clip(bbox[[0, 2]]/self.opt.down_ratio, 0, output_w)
			bbox[[1, 3]] = np.clip(bbox[[1, 3]]/self.opt.down_ratio, 0, output_h)
			h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
			if h > 0 and w > 0:
				radius = gaussian_radius((math.ceil(h), math.ceil(w)))
				radius = max(0, int(radius))
				radius = self.opt.hm_gauss if self.opt.mse_loss else radius
				ct = np.array(
					[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
				ct_int = ct.astype(np.int32)
				draw_gaussian(hm[cls_id], ct_int, radius)
				wh[k] = 1. * w, 1. * h
				ind[k] = ct_int[1] * output_w + ct_int[0]
				reg[k] = ct - ct_int
				reg_mask[k] = 1
				cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
				cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
				if self.opt.dense_wh:
					draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)  # 根据高斯分布设置hw,多个目标取高斯分布大的值覆盖
				gt_det[k]=np.array([ct[0] - w / 2, ct[1] - h / 2,ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

		ret = {'img': img, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}#所有的长度都是在output的heatmap上的长度
		if self.opt.dense_wh:
			hm_a = hm.max(axis=0, keepdims=True)
			dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
			ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
			del ret['wh']
		elif self.opt.cat_spec_wh:
			ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
			del ret['wh']
		if self.opt.reg_offset:
			ret.update({'reg': reg})
		if self.opt.visual > 0 or not self.split == 'train':
			ret['gt'] = gt_det
		return ret