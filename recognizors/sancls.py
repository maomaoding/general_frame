from .base_recognizor import BaseRecognizor
from datasets import get_dataset
from utils.utils import AverageMeter
import torch
import tqdm

def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
	# 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
	assert (output.dim() in [1, 2, 3])
	assert output.shape == target.shape
	output = output.view(-1)
	target = target.view(-1)
	output[target == ignore_index] = ignore_index
	intersection = output[output == target]
	# https://github.com/pytorch/pytorch/issues/1382
	area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
	area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
	area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
	area_union = area_output + area_target - area_intersection
	return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

class SANcls(BaseRecognizor):
	def prepare_input(self, image):
		resized_image = cv2.resize(image, (self.opt.input_w, self.opt.input_h), interpolation=cv2.INTER_AREA)
		inp_image = ((resized_image / 255. - self.mean) / self.std).astype(np.float32)
		inp_image = inp_image.transpose(2, 0, 1).reshape(1, 3, self.opt.input_h, self.opt.input_w)
		inp_image = torch.from_numpy(inp_image)
		return inp_image

	def process(self, image):
		output = self.model(image)
		forward_time = time.time()

		return output, output, forward_time

	def val_metric(self):
		val_dataset = get_dataset(self.opt, "val")
		val_loader = torch.utils.data.DataLoader(
			val_dataset,
			batch_size=6,
			shuffle=False,
			num_workers=1,
			pin_memory=True,
			collate_fn=val_dataset.collate_fn if 'collate_fn' in dir(val_dataset) else None,
		)

		top1 = AverageMeter()
		top5 = AverageMeter()
		intersection_meter = AverageMeter()
		union_meter = AverageMeter()
		target_meter = AverageMeter()

		for batch in tqdm.tqdm(val_loader):
			input, target = batch['img'], batch['annot']
			input = input.to(self.opt.device)
			target = target.to(self.opt.device)
			output = self.model(input)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
			top1.update(acc1.item(), input.size(0))
			top5.update(acc5.item(), input.size(0))

			output = output.max(1)[1]
			intersection, union, target = intersectionAndUnionGPU(output, target, self.opt.num_classes)
			intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
			intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

		accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)

		for i in range(self.opt.num_classes):
			print('Class_{} Result: accuracy {:.4f}.'.format(i, accuracy_class[i]))
		return "top1: {}, top5: {}".format(top1.avg, top5.avg)