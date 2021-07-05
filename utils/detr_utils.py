import torch
from torchvision.ops.boxes import box_area
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if float(torchvision.__version__[:3]) < 0.7:
	from torchvision.ops import _new_empty_tensor
	from torchvision.ops.misc import _output_size

def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(-1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
		(x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
	x0, y0, x1, y1 = x.unbind(-1)
	b = [(x0 + x1) / 2, (y0 + y1) / 2,
		(x1 - x0), (y1 - y0)]
	return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
	area1 = box_area(boxes1)
	area2 = box_area(boxes2)

	lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # [N,M,2]
	rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N,M,2]

	wh = (rb - lt).clamp(min=0) # [N,M,2]
	inter = wh[:, :, 0] * wh[:, :, 1] # [N,M]

	union = area1[:, None] + area2 - inter

	iou = inter / union
	return iou, union

def generalized_box_iou(boxes1, boxes2):
	"""
	Generalized IoU from https://giou.stanford.edu/

	The boxes should be in [x0, y0, x1, y1] format

	Returns a [N, M] pairwise matrix, where N = len(boxes1)
	and M = len(boxes2)
	"""
	# degenerate boxes gives inf / nan results
	# so do an early check
	assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
	assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
	iou, union = box_iou(boxes1, boxes2)

	lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
	rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

	wh = (rb - lt).clamp(min=0) # [N,M,2]
	area = wh[:, :, 0] * wh[:, :, 1]

	return iou - (area - union) / area

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	if target.numel() == 0:
		return [torch.zeros([], device=output.device)]
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def _max_by_axis(the_list):
	# type: (List[List[int]]) -> List[int]
	maxes = the_list[0]
	for sublist in the_list[1:]:
		for index, item in enumerate(sublist):
			maxes[index] = max(maxes[index], item)
	return maxes

def nested_tensor_from_tensor_list(tensor_list):
	if tensor_list[0].ndim == 3:
		if torchvision._is_tracing():
			raise ValueError('not supported')

		# TODO make it support different-sized images
		max_size = _max_by_axis([list(img.shape) for img in tensor_list])
		batch_shape = [len(tensor_list)] + max_size
		b, c, h, w = batch_shape
		dtype = tensor_list[0].dtype
		device = tensor_list[0].device
		tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
		mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
		for img, pad_img, m in zip(tensor_list, tensor, mask):
			pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
			m[: img.shape[1], : img.shape[2]] = False
	else:
		raise ValueError('not supported')
	return NestedTensor(tensor, mask)

class NestedTensor():
	def __init__(self, tensors, mask):
		self.tensors = tensors
		self.mask = mask

	def to(self, device, **args):
		cast_tensor = self.tensors.to(device, **args)
		mask = self.mask
		if mask is not None:
			assert mask is not None
			cast_mask = mask.to(device, **args)
		else:
			cast_mask = None
		return NestedTensor(cast_tensor, cast_mask)

	def size(self, idx):
		return self.tensors.size(idx)

	def decompose(self):
		return self.tensors, self.mask

	def __repr__(self):
		return str(self.tensors)

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
	# type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
	"""
	Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
	This will eventually be supported natively by PyTorch, and this
	class can go away.
	"""
	if float(torchvision.__version__[:3]) < 0.7:
		if input.numel() > 0:
			return torch.nn.functional.interpolate(
				input, size, scale_factor, mode, align_corners
			)

		output_shape = _output_size(2, input, size, scale_factor)
		output_shape = list(input.shape[:-2]) + list(output_shape)
		return _new_empty_tensor(input, output_shape)
	else:
		return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
	"""
	Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
	Args:
		inputs: A float tensor of arbitrary shape.
				The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs
				(0 for the negative class and 1 for the positive class).
		alpha: (optional) Weighting factor in range (0,1) to balance
				positive vs negative examples. Default = -1 (no weighting).
		gamma: Exponent of the modulating factor (1 - p_t) to
			   balance easy vs hard examples.
	Returns:
		Loss tensor
	"""
	prob = inputs.sigmoid()
	ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
	p_t = prob * targets + (1 - prob) * (1 - targets)
	loss = ce_loss * ((1 - p_t) ** gamma)

	if alpha >= 0:
		alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
		loss = alpha_t * loss

	return loss.mean(1).sum() / num_boxes

def dice_loss(inputs, targets, num_boxes):
	"""
	Compute the DICE loss, similar to generalized IOU for masks
	Args:
		inputs: A float tensor of arbitrary shape.
				The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs
				(0 for the negative class and 1 for the positive class).
	"""
	inputs = inputs.sigmoid()
	inputs = inputs.flatten(1)
	numerator = 2 * (inputs * targets).sum(1)
	denominator = inputs.sum(-1) + targets.sum(-1)
	loss = 1 - (numerator + 1) / (denominator + 1)
	return loss.sum() / num_boxes