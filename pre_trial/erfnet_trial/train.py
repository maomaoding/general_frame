from dataset import *
from model import *
from utils import *
import math,shutil
import lovasz_losses as L

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = None
		self.avg = None
		self.sum = None
		self.count = None

	def update(self, val, n=1):
		if self.val is None:
			self.val = val
			self.sum = val * n
			self.count = n
			self.avg = self.sum / self.count
		else:
			self.val = val
			self.sum += val * n
			self.count += n
			self.avg = self.sum / self.count
class EvalSegmentation(object):
	def __init__(self, num_class, ignore_label=None):
		self.num_class = num_class
		self.ignore_label = ignore_label

	def __call__(self, pred, gt):
		assert (pred.shape == gt.shape)
		gt = gt.flatten().astype(int)
		pred = pred.flatten().astype(int)
		locs = (gt != self.ignore_label)
		sumim = gt + pred * self.num_class
		hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
		return hs

class Trainer():
	def __init__(self):
		self.max_epoch = 60
		self.best_pred = 0.0
		self.metric = SegmentationMetric(5)
		self.mean = (.485, .456, .406)
		self.std = (.229, .224, .225)
		self.train_dataset = culane(mode='train', mean=self.mean, std=self.std)
		self.val_dataset = culane(mode='val', mean=self.mean, std=self.std)

		self.train_loader = data.DataLoader(dataset=self.train_dataset,
											batch_size=4,
											shuffle=True,
											drop_last=True,
											num_workers=8)
		self.val_loader = data.DataLoader(dataset=self.val_dataset,
										batch_size=1,
										shuffle=False,
										num_workers=8)

		self.max_iters = len(self.train_loader) * self.max_epoch

		self.model = get_erfnet(pretrained=True, pretrained_weights='./snapshot/erfnet_curbstonedata.pth')

		# if torch.cuda.device_count() > 1:
		#  	self.model = torch.nn.DataParallel(self.model, device_ids=[0,1])
		self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

		self.criterion = L.lovasz_softmax
		self.criterion_lanecls = torch.nn.CrossEntropyLoss().to(self.device)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

		# self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
		# 				lambda epoch: math.pow(1-epoch/self.max_iters, 0.9))
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
		# self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=30)

	def train(self):
		for epoch in range(0, self.max_epoch):
			self.model.train()

			for i, (images, targets, laneclss) in enumerate(self.train_loader):

				images = images.to(self.device)
				targets = targets.to(self.device)
				laneclss = laneclss.to(self.device)

				decoder_output = self.model(images)

				# segs_prob = F.softmax(decoder_output, dim=1)
				# seg_loss = self.criterion(segs_prob, targets, ignore=255)

				#cross entropy
				seg_loss = F.cross_entropy(decoder_output, targets, 
					weight=torch.tensor([0.4, 1., 1., 1., 1.]).to(self.device))
				#cross entropy

				# lanecls_loss = self.criterion_lanecls(pred_exists, laneclss)
				# loss = seg_loss + 0.1 * lanecls_loss
				loss = seg_loss

				self.scheduler.optimizer.zero_grad()
				loss.backward()
				self.scheduler.optimizer.step()

				if i % 1 == 0:
					print('Epoch: [{}/{}] Iter [{}/{}] || lr: {} || Loss: {}'.format(
								epoch, self.max_epoch, i+1, len(self.train_loader), 
								self.scheduler.optimizer.param_groups[0]['lr'], loss.item()))

			self.scheduler.step()

			self.validation(epoch)

		# save_checkpoint(model, is_best)

	def validation(self, epoch):
		# IoU = AverageMeter()
		is_best = False
		self.metric.reset()
		self.model.eval()
		for i, (image, target, exists) in enumerate(self.val_loader):
			image = image.to(self.device)
			target = target.to(self.device)

			decoder_output = self.model(image)

			segs_prob = F.softmax(decoder_output, dim=1)
			seg_loss = self.criterion(segs_prob, target, ignore=255)
			target = target.cpu()

			pred_seg = torch.argmax(decoder_output, 1)
			pred_seg = pred_seg.cpu().data.numpy()

			# evaluator = EvalSegmentation(5, 255)
			# IoU.update(evaluator(pred_seg, target.numpy()))
			# mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
			# mIoU = np.sum(mIoU) / len(mIoU)

			self.metric.update(pred_seg, target.numpy())
			pixAcc, mIoU = self.metric.get()
			print('Epoch %d, Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%, segloss: %.3f%%' % (
				epoch, i + 1, pixAcc * 100, mIoU * 100, seg_loss))

		# new_pred = (pixAcc + mIoU) / 2
		new_pred = mIoU
		if new_pred > self.best_pred:
			is_best = True
			self.best_pred = new_pred
		self.save_checkpoint(is_best)

	def save_checkpoint(self,is_best=False):
		"""Save Checkpoint"""
		directory = os.path.expanduser('./snapshot')
		if not os.path.exists(directory):
			os.makedirs(directory)
		filename = '{}_{}.pth'.format('erfnet', 'curbstonedata')
		save_path = os.path.join(directory, filename)
		torch.save(self.model.state_dict(), save_path)
		if is_best:
			best_filename = '{}_{}_best_model.pth'.format('erfnet', 'curbstonedata')
			best_filename = os.path.join(directory, best_filename)
			shutil.copyfile(save_path, best_filename)

if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()
	# trainer.validation(1)