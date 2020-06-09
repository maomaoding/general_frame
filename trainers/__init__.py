from .centernet_trainer import CenterNetTrainer
from .unetpp_trainer import UnetppTrainer
from .erfnet_trainer import erfnetTrainer
from .efficientdet_trainer import EfficientDetTrainer

_train_factory = {
	'centernet': CenterNetTrainer,
	'unetppseg': UnetppTrainer,
	'erfnetseg': erfnetTrainer,
	'efficientdet': EfficientDetTrainer,
}

def get_trainer(opt):
	return _train_factory[opt.task](opt)