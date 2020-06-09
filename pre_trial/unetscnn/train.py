from dataset import *
from model import *
from utils import *
import math,shutil
import lovasz_losses as L

class Trainer():
	def __init__(self):
		self.max_epoch = 200
		self.best_pred = 0.0
		self.metric = SegmentationMetric(2)
		self.input_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
		])
		self.train_dataset = curbstonedata(mode='train', transform=self.input_transform)
		self.val_dataset = curbstonedata(mode='val', transform=self.input_transform)

		self.train_loader = data.DataLoader(dataset=self.train_dataset,
											batch_size=32,
											shuffle=True,
											drop_last=True)
		self.val_loader = data.DataLoader(dataset=self.val_dataset,
										batch_size=1,
										shuffle=False)

		self.max_iters = len(self.train_loader) * self.max_epoch

		self.model = get_unet(pretrained_weights='./snapshot/scnn_curbstonedata.pth')

		# if torch.cuda.device_count() > 1:
		#  	self.model = torch.nn.DataParallel(self.model, device_ids=[0,1])
		self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

		self.criterion = L.lovasz_softmax

		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-8)

		self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
						lambda epoch: math.pow(1-epoch/self.max_iters, 0.9))

	def train(self):
		for epoch in range(0, self.max_epoch):
			self.model.train()

			for i, (images, targets) in enumerate(self.train_loader):
				images = images.to(self.device)
				targets = targets.to(self.device)

				output = self.model(images)

				output_prob = F.softmax(output, dim=1)
				loss = self.criterion(output_prob, targets, ignore=255)

				#cross entropy
				# loss = F.cross_entropy(output, targets, 
				# 	weight=torch.tensor([0.4, 1.]).to(self.device))
				#cross entropy

				self.scheduler.optimizer.zero_grad()
				loss.backward()
				self.scheduler.optimizer.step()
				self.scheduler.step()

				if i % 10 == 0:
					print('Epoch: [{}/{}] Iter [{}/{}] || lr: {} || Loss: {}'.format(
								epoch, self.max_epoch, i+1, len(self.train_loader), 
								self.scheduler.optimizer.param_groups[0]['lr'], loss.item()))

			self.validation(epoch)

		# save_checkpoint(model, is_best)

	def validation(self, epoch):
		is_best = False
		self.metric.reset()
		self.model.eval()
		for i, (image, target) in enumerate(self.val_loader):
			image = image.to(self.device)

			outputs = self.model(image)
			pred = torch.argmax(outputs, 1)
			pred = pred.cpu().data.numpy()
			self.metric.update(pred, target.numpy())
			pixAcc, mIoU = self.metric.get()
			print('Epoch %d, Sample %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' % (
				epoch, i + 1, pixAcc * 100, mIoU * 100))

		new_pred = (pixAcc + mIoU) / 2
		if new_pred > self.best_pred:
			is_best = True
			self.best_pred = new_pred
		self.save_checkpoint(is_best)

	def save_checkpoint(self,is_best=False):
		"""Save Checkpoint"""
		directory = os.path.expanduser('./snapshot')
		if not os.path.exists(directory):
			os.makedirs(directory)
		filename = '{}_{}.pth'.format('scnn', 'curbstonedata')
		save_path = os.path.join(directory, filename)
		torch.save(self.model.state_dict(), save_path)
		if is_best:
			best_filename = '{}_{}_best_model.pth'.format('scnn', 'curbstonedata')
			best_filename = os.path.join(directory, best_filename)
			shutil.copyfile(save_path, best_filename)

if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()
