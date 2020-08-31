model = dict(
	task = 'efficientdet',
	arch = 'efficientdet',
	model_path = "./checkpoints/model_efficientdetbest.pth",
	compound_coef = 1,

	load_weights = False,
	ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
	scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
	)

dataset = dict(
	data_dir="/home/dingyaohua/remote/datasets",
	dataset="junctions",
	keep_res=False, #保持分辨率
	mean = [0.40789654, 0.44719302, 0.47026115],
	std = [0.28863828, 0.27408164, 0.27809835],
	num_classes = 30,

	pad = 31,
	input_h = 640,
	input_w = 640,

	class_name=['corner_right_bottom', 'corner_left_bottom', 'corner_left_top', 'corner_right_top',
				'cross_top', 'cross_right', 'cross_bottom', 'cross_left', 'cross_all',
				'rectangle_top', 'rectangle_right', 'rectangle_bottom', 'rectangle_left',
				'door_top', 'door_right', 'door_bottom', 'door_left',
				'horizontal_sliding_door', 'vertical_sliding_door',
				'double_door_top', 'double_door_right', 'double_door_bottom', 'double_door_left',
				'horizontal_window', 'vertical_window',
				'bay_window_top', 'bay_window_right', 'bay_window_bottom', 'bay_window_left',
				'flue']
	)

train_cfg=dict(
	#train
	batch_size=6,
	lr = 1e-4,
	)

visual=False
show_results=False
export_onnx = False