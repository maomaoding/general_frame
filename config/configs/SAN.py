model = dict(
	task='SAN',
	arch='san',
	model_path = "./checkpoints/pretrained_weights_san.pth",
	)

dataset = dict(
	data_dir="/home/shejiyun/DingYaohua/datasets",
	# data_dir="/home/dingyaohua/remote/datasets",
	dataset="furnitures",

	class_map='',
	# keep_res=False, #保持分辨率
	mean = [0.40789654, 0.44719302, 0.47026115],
	std = [0.28863828, 0.27408164, 0.27809835],
	num_classes=54,

	input_h = 224,
	input_w = 224,
	)

train_cfg = dict(
	#mixup
	mixup=0.0,
	cutmix=0.0,
	cutmix_minmax=None,
	mixup_prob=1.0,
	mixup_switch_prob=0.5,
	mixup_mode='batch',
	smoothing=0.0, #default 0.1
	#mixup

	sa_type=1,
	layers=[2,1,2,4,1],
	kernels=[3,7,7,7,7],

	batch_size=4,
	num_workers=1,
	)

visual=False
export_onnx=False