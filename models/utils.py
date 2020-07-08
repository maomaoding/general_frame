import torch,os
import torch.nn as nn
import math
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

def variance_scaling_(tensor, gain=1.):
	# type: (Tensor, float) -> Tensor
	r"""
	initializer for SeparableConv in Regressor/Classifier
	reference: https://keras.io/zh/initializers/  VarianceScaling
	"""
	fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
	std = math.sqrt(gain / float(fan_in))

	return _no_grad_normal_(tensor, 0., std)

def init_weights(model):
	for name, module in model.named_modules():
		is_conv_layer = isinstance(module, nn.Conv2d)

		if is_conv_layer:
			if "conv_list" or "header" in name:
				variance_scaling_(module.weight.data)
			else:
				nn.init.kaiming_uniform_(module.weight.data)

			if module.bias is not None:
				if "classifier.header" in name:
					bias_value = -np.log((1 - 0.01) / 0.01)
					torch.nn.init.constant_(module.bias, bias_value)
				else:
					module.bias.data.zero_()

def load_model(model, model_path, optimizer=None, resume=False):
	start_epoch = 0
	checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
	print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
	state_dict_ = checkpoint['state_dict']
	state_dict = {}

	# convert data_parallal to model
	for k in state_dict_:
		if k.startswith('module') and not k.startswith('module_list'):
			state_dict[k[7:]] = state_dict_[k]
		else:
			state_dict[k] = state_dict_[k]
	model_state_dict = model.state_dict()

	# check loaded parameters and created model parameters
	msg = 'If you see this, your model does not fully load the ' + \
			'pre-trained weight. Please make sure ' + \
			'you have correctly specified --arch xxx ' + \
			'or set the correct --num_classes for your own dataset.'
	for k in state_dict:
		if k in model_state_dict:
			if state_dict[k].shape != model_state_dict[k].shape:
				print('Skip loading parameter {}, required shape{}, ' \
						'loaded shape{}. {}'.format(
					k, model_state_dict[k].shape, state_dict[k].shape, msg))
				state_dict[k] = model_state_dict[k]
		else:
			print('Drop parameter {}.'.format(k) + msg)
	for k in model_state_dict:
		if not (k in state_dict):
			print('No param {}.'.format(k) + msg)
			state_dict[k] = model_state_dict[k]
	model.load_state_dict(state_dict, strict=False)

	# resume optimizer parameters
	if optimizer is not None and resume:
		if 'optimizer' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer'])
			start_epoch = checkpoint['epoch']
			start_lr = checkpoint['lr']
			for param_group in optimizer.param_groups:
				param_group['lr'] = start_lr
			print('Resumed optimizer with start lr', start_lr)
		else:
			print('No optimizer parameters in checkpoint.')
	if optimizer is not None:
		return model, optimizer, start_epoch
	else:
		return model

def save_model(path, epoch, model, optimizer=None):
	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))
	if isinstance(model, torch.nn.DataParallel):
		state_dict = model.module.state_dict()
	else:
		state_dict = model.state_dict()
	data = {'epoch': epoch,
			'state_dict': state_dict,
			'lr': optimizer.param_groups[-1]['lr']}
	if not (optimizer is None):
		data['optimizer'] = optimizer.state_dict()
	torch.save(data, path)