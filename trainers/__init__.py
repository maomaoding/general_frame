from .centernet_trainer import CenterNetTrainer
from .unetpp_trainer import UnetppTrainer
from .erfnet_trainer import erfnetTrainer

_train_factory = {
	'centernet': CenterNetTrainer,
	'unetppseg': UnetppTrainer,
	'erfnetseg': erfnetTrainer,
}

def get_trainer(opt):
	return _train_factory[opt.task](opt)