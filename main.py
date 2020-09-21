from config.opts import opts
from datasets import get_dataset
from trainers import get_trainer
from det_cls_ors import get_det_cls_ors
from models.utils import save_model
from utils.visualizer import Visualizer
from utils.flops_counter import get_model_complexity_info
import torch,os,cv2,json,fire
import numpy as np

def test_imgfolder():
	time_stats = ['tot', 'pre', 'net']
	opt = opts()
	opt.from_file('./config/configs/efficientdet.py')
	det_cls_ors = get_det_cls_ors(opt)
	det_cls_ors.pause=False
	opt.show_results=False

	output_folder = './output'
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	folder_path = '/home/dingyaohua/Downloads/jira_bug_png'
	for file in os.listdir(folder_path):
		if 'txt' in file:
			continue
		img = cv2.imread(os.path.join(folder_path, file))

		ret = det_cls_ors.run(img)
		cv2.imwrite(os.path.join(output_folder, file), ret['dets_image'])

		time_str = ''
		for stat in time_stats:
			time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
		print(time_str)

def output_model_flops():
	opt = opts()
	opt.from_file('./config/configs/centernet.py')
	det_cls_ors = get_det_cls_ors(opt)
	flops, params = get_model_complexity_info(det_cls_ors.model.cuda(), (3,640,640), as_strings=True,
											print_per_layer_stat=True)
	print('Params/Flops: {}/{}'.format(params, flops))

def output_model_weights():
	opt = opts()
	opt.from_file('./config/configs/centernet.py')
	det_cls_ors = get_det_cls_ors(opt)

	torch.save(det_cls_ors.model.state_dict(), 'centernet_ver7_640.pth')

def test_video():
	time_stats = ['tot', 'pre', 'net']
	opt = opts()
	opt.from_file('./config/configs/centernet.py')
	det_cls_ors = get_det_cls_ors(opt)
	det_cls_ors.pause=False

	video_path = '/home/cowa/data_server/zhq/2019-09-10_16.59/_camera_image_raw.mp4'
	cam = cv2.VideoCapture(video_path)
	cam.set(cv2.CAP_PROP_POS_FRAMES, 10000)
	while True:
		_, img = cam.read()
		ret = det_cls_ors.run(img)
		time_str = ''
		for stat in time_stats:
			time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
		print(time_str)

def val():
	opt = opts()
	opt.from_file('./config/configs/SAN.py')
	det_cls_ors = get_det_cls_ors(opt)
	ap = det_cls_ors.val_metric()
	print(ap)

def train():
	opt = opts()
	opt.from_file('./config/configs/SAN.py')

	print('Setting up data...')
	dataset=get_dataset(opt,"train")
	train_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.num_workers,
		pin_memory=True,
		drop_last=True,
		collate_fn=dataset.collate_fn if 'collate_fn' in dir(dataset) else None,
	)
	val_loader = torch.utils.data.DataLoader(
		get_dataset(opt,"val"),
		batch_size=1,
		shuffle=False,
		num_workers=1,
		pin_memory=True,
		collate_fn=dataset.collate_fn if 'collate_fn' in dir(dataset) else None,
	)

	print('Creating model...')
	trainer = get_trainer(opt)
	model=trainer.get_model()
	optimizer=trainer.get_optimizer()
	print('Starting training...')
	best = 1e10
	start_epoch = 0

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

	for epoch in range(start_epoch + 1, opt.num_epochs + 1):
		log_dict_train = trainer.train(epoch, train_loader)

		with torch.no_grad():
			log_dict_val = trainer.val(epoch, val_loader)

		if log_dict_val["loss"] < best:
			best = log_dict_val["loss"]
			save_model(os.path.join(opt.save_dir, 'model_{}best.pth'.format(opt.arch)),
					   epoch, model, optimizer)
		# if log_dict_train["loss"] < best:
		# 	best = log_dict_train["loss"]
		# 	save_model(os.path.join(opt.save_dir, 'model_{}best.pth'.format(opt.dataset)),
		# 			   epoch, model, optimizer)

		scheduler.step()
		print('Drop LR to', scheduler.optimizer.param_groups[-1]['lr'])
	trainer.writer.close()

if __name__ == '__main__':
	fire.Fire()
	# train()
	# test_video()
	# val()
	# test_imgfolder()
