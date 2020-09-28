model = dict(
	task = 'erfnet',
	model_path = "",
	num_classes = 8,
	num_labels = 4,
	model_onnx_path="erfnet.onnx",
	export_onnx=False, #是否导出onnx模型,train时无效，test时在base_detector中传入
	)

dataset = dict(
	# data_dir = '/datastore/data/dataset/COWA/',
	data_dir = '/datastore/data/dataset/',
	# dataset = 'lane',
	dataset = 'CULane',
	mean = [126.519748/255.0,131.187286/255.0,127.388270/255.0],
	std = [11.248100/255.0,11.453702/255.0,11.286641/255.0],
	)

data_augment = [{'name': 'addGaussianNoise_seg'},
				{'name': 'flipHorizon_seg'},
				{'name': 'randomCrop_seg'},
				{'name': 'Resize_seg', 'resize_width': 512, 'resize_height': 256},
				{'name': 'Normalize_seg', 'mean': dataset['mean'], 'std': dataset['std']}]

train_cfg = dict(
	batch_size = 4,
	# lr = 0.00001,
	# num_workers = 1,
	loss_weight = [1, 0.0],
	lane_weight = [0.4, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0],
	optimizer={'name': 'Adam', 'weight_decay': 5e-5},
	)

visual = True