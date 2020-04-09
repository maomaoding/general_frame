model=dict(
	task='centernet',
	arch="dla34",
	model_path = "./checkpoints/model_TT100Kbest.pth",
	head_conv=256,#conv layer channels for output head 0 for no conv layer -1 for default setting:256 for dla
	down_ratio=4,
	reg_offset=True,
	cat_spec_wh=False,
	dense_wh=True, #根据高斯分布设置wh的值
	norm_wh=True,

	model_onnx_path="centernet.onnx",
	export_onnx=False, #是否导出onnx模型,train时无效，test时在base_detector中传入
	)

dataset=dict(
	data_dir="/datastore/data/dataset",
	dataset="TT100K",  #bdd100k  TT100K
	keep_res=False, #保持分辨率
	mean = [0.40789654, 0.44719302, 0.47026115],
	std = [0.28863828, 0.27408164, 0.27809835],
	num_classes=1,
	#augment
	shift_scale_prob=0.7,
	shift_range = 0.1,  # 图像增强位移范围
	scale_range = 0.4,  # 图像增强缩放范围
	contrast_bright_prob=0.7,
	# input 网络输入图像大小
	pad=31, #有上采样的层就需要保证输入大小能被32整除，input_res=1时计算使用
	input_res=-1, #-1：数据集默认值，会被input_h和input_w覆盖
	input_h=512, #input height. -1 for default from dataset
	input_w=720, #input width. -1 for default from dataset
	max_objs = 128,#max number of output objects
	nms=False,

	class_name=[
			'traffic sign', ]
	)

heads = {'hm': dataset['num_classes'],
		 'wh': 2 if not model["cat_spec_wh"] else 2 * dataset['num_classes']}
if model["reg_offset"]:
	heads.update({'reg': 2})
get_heads=dict(
	heads=heads,
	)

train_cfg=dict(
	#loss
	reg_loss='sl1', #'regression loss: sl1 | l1 | l2'
	mse_loss=False,
	hm_gauss=3,
	hm_weight=1,
	off_weight=1,
	wh_weight=0.1,
	#train
	batch_size=8,
	num_iters=-1,# 'default: #samples / batch_size.'
	optimizer={'name': 'Adam', 'weight_decay': 5e-5},
	)

test_cfg=dict(
	#test
	test_scales=[1], #multi scale test augmentation
	val_filepath='/datastore/data/dataset/TT100K/data/test',
	)

data_augment = [{'name': 'RandomCrop', 'random_ratio': 0.5, 'shift': 0.1},
				{'name': 'RandomGasussBlur', 'random_ratio': 0.3},
				{'name': 'RandomContrastBright', 'random_ratio': 0.3},
				{'name': 'RandomNoise', 'random_ratio': 0.3},
				{'name': 'RandomFlip', 'random_ratio': 0.5},
				{'name': 'Normalize', 'mean': dataset['mean'], 'std': dataset['std']}]

vis_thresh=0.3
visual=True
show_results=True