model=dict(
	task='centernet',
	arch="dla34",
	model_path = "./checkpoints/model_junctionsbest.pth",
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
	data_dir="/home/dingyaohua/datasets",
	dataset="junctions",
	keep_res=False, #保持分辨率
	mean = [0.40789654, 0.44719302, 0.47026115],
	std = [0.28863828, 0.27408164, 0.27809835],
	num_classes=29,
	#augment
	shift_scale_prob=0.7,
	shift_range = 0.1,  # 图像增强位移范围
	scale_range = 0.4,  # 图像增强缩放范围
	contrast_bright_prob=0.7,
	# input 网络输入图像大小
	pad=31, #有上采样的层就需要保证输入大小能被32整除，input_res=1时计算使用
	input_res=-1, #-1：数据集默认值，会被input_h和input_w覆盖
	input_h=512, #input height. -1 for default from dataset
	input_w=512, #input width. -1 for default from dataset
	max_objs = 128,#max number of output objects
	nms=False,

	class_name=['corner_right_bottom', 'corner_left_bottom', 'corner_left_top', 'corner_right_top',
				'cross_top', 'cross_right', 'cross_bottom', 'cross_left', 'cross_all',
				'rectangle_top', 'rectangle_right', 'rectangle_bottom', 'rectangle_left',
				'door_top', 'door_right', 'door_bottom', 'door_left',
				'horizontal_sliding_door', 'vertical_sliding_door',
				'double_door_top', 'double_door_right', 'double_door_bottom', 'double_door_left',
				'horizontal_window', 'vertical_window',
				'bay_window_top', 'bay_window_right', 'bay_window_bottom', 'bay_window_left']
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
	batch_size=4,
	num_iters=-1,# 'default: #samples / batch_size.'
	)

test_cfg=dict(
	#test
	test_scales=[1], #multi scale test augmentation
	val_filepath='/datastore/data/dataset/TT100K/data/test',
	)

vis_thresh=0.3
visual=True
show_results=True