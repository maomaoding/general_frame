model = dict(
	task = 'detr',
	model_path = "./checkpoints/detr-r50-e632da11.pth",
	hidden_dim = 256,
	position_embedding = 'sine',#choices=('sine','learned') Type of positional embedding to use on top of the image features
	lr_backbone = 1e-5,
	masks = False,
	backbone = 'resnet50',
	dilation = False,
	dropout = 0.1,
	nheads = 8,
	dim_feedforward = 2048,
	enc_layers = 6,
	dec_layers = 6,
	pre_norm = False,
	num_classes = 31,
	num_queries = 150,
	aux_loss = True,
	set_cost_class = 1,
	set_cost_bbox = 5,
	set_cost_giou = 2,
	bbox_loss_coef = 5,
	giou_loss_coef = 2,
	eos_coef = 0.1,
	weight_decay = 1e-4,
)

dataset = dict(
	data_dir = "/home/dingyaohua/remote/datasets",
	dataset = "user_junctions",
	keep_res=False, #保持分辨率
	mean = [0.40789654, 0.44719302, 0.47026115],
	std = [0.28863828, 0.27408164, 0.27809835],

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

visual = False