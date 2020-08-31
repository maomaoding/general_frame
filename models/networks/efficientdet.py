import torch
import itertools
import numpy as np
from torch import nn
from .base_models.efficientnet_utils import MemoryEfficientSwish, Swish,\
											Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from .base_models.efficientnet import EfficientNet
from torchvision.ops.boxes import batched_nms

def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out

class BBoxTransform(nn.Module):
	def forward(self, anchors, regression):
		"""
		decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

		Args:
			anchors: [batchsize, boxes, (y1, x1, y2, x2)]
			regression: [batchsize, boxes, (dy, dx, dh, dw)]

		Returns:

		"""
		y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
		x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
		ha = anchors[..., 2] - anchors[..., 0]
		wa = anchors[..., 3] - anchors[..., 1]

		w = regression[..., 3].exp() * wa
		h = regression[..., 2].exp() * ha

		y_centers = regression[..., 0] * ha + y_centers_a
		x_centers = regression[..., 1] * wa + x_centers_a

		ymin = y_centers - h / 2.
		xmin = x_centers - w / 2.
		ymax = y_centers + h / 2.
		xmax = x_centers + w / 2.

		return torch.stack([xmin, ymin, xmax, ymax], dim=2)

class ClipBoxes(nn.Module):

	def __init__(self):
		super(ClipBoxes, self).__init__()

	def forward(self, boxes, img):
		batch_size, num_channels, height, width = img.shape

		boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
		boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

		boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
		boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

		return boxes

class Anchors(nn.Module):
	def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
		super().__init__()
		self.anchor_scale = anchor_scale

		if pyramid_levels is None:
			self.pyramid_levels = [3,4,5,6,7]

		self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
		self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
		self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

		self.last_anchors = {}
		self.last_shape = None

	def forward(self, image, dtype=torch.float32):
		image_shape = image.shape[2:]

		if image_shape == self.last_shape and image.device in self.last_anchors:
			return self.last_anchors[image.device]

		if self.last_shape is None or self.last_shape != image_shape:
			self.last_shape = image_shape

		if dtype == torch.float16:
			dtype = np.float16
		else:
			dtype = np.float32

		boxes_all = []
		for stride in self.strides:
			boxes_level = []
			for scale, ratio in itertools.product(self.scales, self.ratios):
				if image_shape[1] % stride != 0:
					raise ValueError('input size must be divided by the stride.')
				base_anchor_size = self.anchor_scale * stride * scale
				anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
				anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

				x = np.arange(stride / 2, image_shape[1], stride)
				y = np.arange(stride / 2, image_shape[0], stride)
				xv, yv = np.meshgrid(x, y)
				xv = xv.reshape(-1)
				yv = yv.reshape(-1)

				# y1,x1,y2,x2
				boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
									yv + anchor_size_y_2, xv + anchor_size_x_2))
				boxes = np.swapaxes(boxes, 0, 1)
				boxes_level.append(np.expand_dims(boxes, axis=1))
			# concat anchors on the same level to the reshape NxAx4
			boxes_level = np.concatenate(boxes_level, axis=1)
			boxes_all.append(boxes_level.reshape([-1, 4]))

		anchor_boxes = np.vstack(boxes_all)

		anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
		anchor_boxes = anchor_boxes.unsqueeze(0)

		# save it for later use to reduce overhead
		self.last_anchors[image.device] = anchor_boxes
		return anchor_boxes


class SeparableConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
		super(SeparableConvBlock, self).__init__()
		if out_channels == None:
			out_channels = in_channels

		self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
													kernel_size=3, stride=1, groups=in_channels, bias=False)
		self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

		self.norm = norm
		if self.norm:
			self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

		self.activation = activation
		if self.activation:
			self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

	def forward(self, x):
		x = self.depthwise_conv(x)
		x = self.pointwise_conv(x)

		if self.norm:
			x = self.bn(x)
		if self.activation:
			x = self.swish(x)

		return x


class BiFPN(nn.Module):
	def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
		super(BiFPN, self).__init__()
		self.epsilon = epsilon
		#Conv layers
		self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
		self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
		self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
		self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
		self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
		self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
		self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
		self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

		# Feature scaling layers
		self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

		self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
		self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
		self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
		self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

		self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

		self.first_time = first_time
		if self.first_time:
			self.p5_down_channel = nn.Sequential(
				Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
				nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
			)
			self.p4_down_channel = nn.Sequential(
				Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
				nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
			)
			self.p3_down_channel = nn.Sequential(
				Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
				nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
			)

			self.p5_to_p6 = nn.Sequential(
				Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
				nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
				MaxPool2dStaticSamePadding(3, 2)
			)
			self.p6_to_p7 = nn.Sequential(
				MaxPool2dStaticSamePadding(3, 2)
			)

			self.p4_down_channel_2 = nn.Sequential(
				Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
				nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
			)
			self.p5_down_channel_2 = nn.Sequential(
				Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
				nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
			)

		# Weight
		self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
		self.p6_w1_relu = nn.ReLU()
		self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
		self.p5_w1_relu = nn.ReLU()
		self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
		self.p4_w1_relu = nn.ReLU()
		self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
		self.p3_w1_relu = nn.ReLU()

		self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
		self.p4_w2_relu = nn.ReLU()
		self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
		self.p5_w2_relu = nn.ReLU()
		self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
		self.p6_w2_relu = nn.ReLU()
		self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
		self.p7_w2_relu = nn.ReLU()

		self.attention = attention

	def forward(self, inputs):
		"""
		illustration of a minimal bifpn unit
			P7_0 -------------------------> P7_2 -------->
			   |-------------|                ↑
							 ↓                |
			P6_0 ---------> P6_1 ---------> P6_2 -------->
			   |-------------|--------------↑ ↑
							 ↓                |
			P5_0 ---------> P5_1 ---------> P5_2 -------->
			   |-------------|--------------↑ ↑
							 ↓                |
			P4_0 ---------> P4_1 ---------> P4_2 -------->
			   |-------------|--------------↑ ↑
							 |--------------↓ |
			P3_0 -------------------------> P3_2 -------->
		"""

		# downsample channels using same-padding conv2d to target phase's if not the same
		# judge: same phase as target,
		# if same, pass;
		# elif earlier phase, downsample to target phase's by pooling
		# elif later phase, upsample to target phase's by nearest interpolation
		if self.attention:
			p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
		else:
			p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)
		return p3_out, p4_out, p5_out, p6_out, p7_out

	def _forward_fast_attention(self, inputs):
		if self.first_time:
			p3, p4, p5 = inputs

			p6_in = self.p5_to_p6(p5)
			p7_in = self.p6_to_p7(p6_in)

			p3_in = self.p3_down_channel(p3)
			p4_in = self.p4_down_channel(p4)
			p5_in = self.p5_down_channel(p5)
		else:
			p3_in, p4_in, p5_in, p6_in, p7_in = inputs

		# Weights for P6_0 and P7_0 to P6_1
		p6_w1 = self.p6_w1_relu(self.p6_w1)
		weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
		# Connections for P6_0 and P7_0 to P6_1 respectively
		p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

		# Weights for P5_0 and P6_1 to P5_1
		p5_w1 = self.p5_w1_relu(self.p5_w1)
		weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
		# Connections for P5_0 and P6_1 to P5_1 respectively
		p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

		# Weights for P4_0 and P5_1 to P4_1
		p4_w1 = self.p4_w1_relu(self.p4_w1)
		weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
		# Connections for P4_0 and P5_1 to P4_1 respectively
		p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

		# Weights for P3_0 and P4_1 to P3_2
		p3_w1 = self.p3_w1_relu(self.p3_w1)
		weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
		# Connections for P3_0 and P4_1 to P3_2 respectively
		p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

		if self.first_time:
			p4_in = self.p4_down_channel_2(p4)
			p5_in = self.p5_down_channel_2(p5)

		# Weights for P4_0, P4_1 and P3_2 to P4_2
		p4_w2 = self.p4_w2_relu(self.p4_w2)
		weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
		# Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
		p4_out = self.conv4_down(
			self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

		# Weights for P5_0, P5_1 and P4_2 to P5_2
		p5_w2 = self.p5_w2_relu(self.p5_w2)
		weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
		# Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
		p5_out = self.conv5_down(
			self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

		# Weights for P6_0, P6_1 and P5_2 to P6_2
		p6_w2 = self.p6_w2_relu(self.p6_w2)
		weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
		# Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
		p6_out = self.conv6_down(
			self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

		# Weights for P7_0 and P6_2 to P7_2
		p7_w2 = self.p7_w2_relu(self.p7_w2)
		weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
		# Connections for P7_0 and P6_2 to P7_2
		p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

		return p3_out, p4_out, p5_out, p6_out, p7_out

	def _forward(self, inputs):
		if self.first_time:
			p3, p4, p5 = inputs

			p6_in = self.p5_to_p6(p5)
			p7_in = self.p6_to_p7(p6_in)

			p3_in = self.p3_down_channel(p3)
			p4_in = self.p4_down_channel(p4)
			p5_in = self.p5_down_channel(p5)
		else:
			p3_in, p4_in, p5_in, p6_in, p7_in = inputs

		# P7_0 to P7_2

		# Connections for P6_0 and P7_0 to P6_1 respectively
		p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

		# Connections for P5_0 and P6_1 to P5_1 respectively
		p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

		# Connections for P4_0 and P5_1 to P4_1 respectively
		p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

		# Connections for P3_0 and P4_1 to P3_2 respectively
		p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

		if self.first_time:
			p4_in = self.p4_down_channel_2(p4)
			p5_in = self.p5_down_channel_2(p5)

		# Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
		p4_out = self.conv4_down(
			self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

		# Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
		p5_out = self.conv5_down(
			self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

		# Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
		p6_out = self.conv6_down(
			self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

		# Connections for P7_0 and P6_2 to P7_2
		p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

		return p3_out, p4_out, p5_out, p6_out, p7_out

class Regressor(nn.Module):
	def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
		super(Regressor, self).__init__()
		self.num_layers = num_layers

		self.conv_list = nn.ModuleList(
			[SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
		self.bn_list = nn.ModuleList(
			[nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
			range(5)])
		self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
		self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

	def forward(self, inputs):
		feats = []
		for feat, bn_list in zip(inputs, self.bn_list):
			for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
				feat = conv(feat)
				feat = bn(feat)
				feat = self.swish(feat)
			feat = self.header(feat)

			feat = feat.permute(0, 2, 3, 1)
			feat = feat.contiguous().view(feat.shape[0], -1, 4)

			feats.append(feat)

		feats = torch.cat(feats, dim=1)
		return feats

class Classifier(nn.Module):
	def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
		super(Classifier, self).__init__()
		self.num_anchors = num_anchors
		self.num_classes = num_classes
		self.num_layers = num_layers
		self.conv_list = nn.ModuleList(
			[SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
		self.bn_list = nn.ModuleList(
			[nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in 
			range(5)])
		self.header1 = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
		self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

	def forward(self, inputs):
		feats = []
		for feat, bn_list in zip(inputs, self.bn_list):
			for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
				feat = conv(feat)
				feat = bn(feat)
				feat = self.swish(feat)
			feat = self.header1(feat)

			feat = feat.permute(0, 2, 3, 1)
			feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
											self.num_classes)
			feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)
			feats.append(feat)
		feats = torch.cat(feats, dim=1)
		feats = feats.sigmoid()
		return feats

class EfficientDetBackbone(nn.Module):
	def __init__(self, num_classes=29, compound_coef=0, load_weights=False, **kwargs):
		super(EfficientDetBackbone, self).__init__()
		self.compound_coef = compound_coef

		self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
		self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
		self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
		self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
		self.anchor_scale = [1., 1., 4., 4., 4., 4., 4., 5.]
		self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
		self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
		conv_channel_coef = {
			# the channels of P3/P4/P5.
			0: [40, 112, 320],
			1: [40, 112, 320],
			2: [48, 120, 352],
			3: [48, 136, 384],
			4: [56, 160, 448],
			5: [64, 176, 512],
			6: [72, 200, 576],
			7: [72, 200, 576],
		}

		num_anchors = len(self.aspect_ratios) * self.num_scales

		self.bifpn = nn.Sequential(
			*[BiFPN(self.fpn_num_filters[self.compound_coef],
					conv_channel_coef[compound_coef],
					True if _ == 0 else False,
					attention=True if compound_coef < 6 else False)
				for _ in range(self.fpn_cell_repeats[compound_coef])])

		self.num_classes = num_classes
		self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
								   num_layers=self.box_class_repeats[self.compound_coef])
		self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
									num_classes=num_classes,
									num_layers=self.box_class_repeats[self.compound_coef])

		self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)

		self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

	def forward(self, inputs):
		_, p3, p4, p5 = self.backbone_net(inputs)

		features = (p3, p4, p5)
		features = self.bifpn(features)

		regression = self.regressor(features)
		classification = self.classifier(features)
		anchors = self.anchors(inputs, inputs.dtype)

		return features, regression, classification, anchors

def get_efficientdet(opt):
	return EfficientDetBackbone(num_classes=opt.num_classes, compound_coef=opt.compound_coef,
								load_weights=opt.load_weights, ratios=opt.ratios, scales=opt.scales)