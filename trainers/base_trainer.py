import time,torch,os
import datetime
from utils.data_parallel import DataParallel
from utils.utils import AverageMeter
from progress.bar import Bar
from models import create_model
from models.utils import load_model, init_weights
from tensorboardX import SummaryWriter
from utils.visualizer import Visualizer

class ModelWithLoss(torch.nn.Module):
	def __init__(self, model, loss):
		super(ModelWithLoss, self).__init__()
		self.model = model
		self.loss = loss
	def forward(self, batch):
		outputs = self.model(batch['img'])
		loss, loss_stats = self.loss(outputs, batch)
		return outputs, loss, loss_stats

class BaseTrainer:
	def __init__(self, opt):
		self.opt = opt
		self.writer = SummaryWriter('logs' + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
		self.vis = Visualizer(opt)
		self.loss_stats, self.loss = self._get_losses(opt)

		self.model = create_model(opt)

		self.optimizer = self.gen_optimizer()

		if opt.model_path != '':
			self.model = load_model(self.model, opt.model_path, None, opt.resume)
			# self.model.load_state_dict(torch.load(opt.model_path), strict=False)
		else:
			print('[Info] initializing weights...')
			init_weights(self.model)

		self.model_with_loss = ModelWithLoss(self.model, self.loss)
		self.set_device(opt.gpus, opt.chunk_sizes, opt.device)

	def get_optimizer(self):
		return self.optimizer

	def get_model(self):
		return self.model

	def set_device(self, gpus, chunk_sizes, device):
		if len(gpus) > 1:
			self.model_with_loss = DataParallel(
					self.model_with_loss, device_ids=gpus,
					chunk_sizes=chunk_sizes).to(device)
		else:
			self.model_with_loss = self.model_with_loss.to(device)

		for state in self.optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device=device, non_blocking=True)

	def run_epoch(self, phase, epoch, data_loader):
		model_with_loss = self.model_with_loss
		opt = self.opt
		if phase == 'train':
			model_with_loss.train()
		else:
			if len(opt.gpus) > 1:
				model_with_loss = self.model_with_loss.module
			model_with_loss.eval()
			torch.cuda.empty_cache()
		data_time, batch_time = AverageMeter(), AverageMeter()
		avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
		num_iters = len(data_loader)
		bar = Bar('{}:'.format(opt.task), max=num_iters)
		end = time.time()
		for iter_id, batch in enumerate(data_loader):
			data_time.update(time.time() - end)
			for k in batch:
				if k != 'gt':
					batch[k] = batch[k].to(device=opt.device, non_blocking=True)
			output, loss, loss_stats = model_with_loss(batch)
			# loss = loss.mean()
			if phase == 'train':
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			batch_time.update(time.time() - end)
			end = time.time()

			Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:}'.format(# |ETA: {eta:}
				epoch, iter_id, num_iters, phase=phase,
				total=bar.elapsed_td)#, eta=bar.eta_td
			for l in avg_loss_stats:
				avg_loss_stats[l].update(
					loss_stats[l].mean().item(), batch['img'].size(0))
				Bar.suffix += '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

			Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
									'|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
			if opt.print_iter > 0:
				if iter_id % opt.print_iter == 0:
					print('{}| {}'.format(opt.task, Bar.suffix))
			else:
				bar.next()

			# self.writer.add_scalar('Loss', avg_loss_stats['loss'].avg, (epoch-1)*num_iters+iter_id)
			# self.writer.add_scalar('reg_loss', avg_loss_stats['reg_loss'].avg, (epoch-1)*num_iters+iter_id)
			# self.writer.add_scalar('cls_loss', avg_loss_stats['cls_loss'].avg, (epoch-1)*num_iters+iter_id)

			if opt.visual > 0:
				if opt.print_iter > 0:
					if iter_id % opt.print_iter == 0:
						self.visual(self.vis, batch, output)
			del output, loss, loss_stats

		bar.finish()
		ret = {k: v.avg for k, v in avg_loss_stats.items()}
		ret['time'] = bar.elapsed_td.total_seconds() / 60.
		return ret

	def visual(self, batch, output):
		raise NotImplementedError

	def _get_losses(self, opt):
		raise NotImplementedError

	def val(self, epoch, data_loader):
		return self.run_epoch('val', epoch, data_loader)

	def train(self, epoch, data_loader):
		return self.run_epoch('train', epoch, data_loader)

	def gen_optimizer(self):
		raise NotImplementedError