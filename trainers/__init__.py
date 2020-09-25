from utils.registry import get_trainer
import os,sys,importlib
sys.path.append(os.path.dirname(__file__))
for file in os.listdir(os.path.dirname(__file__)):
	if '.py' in file and 'base' not in file and '__init__' not in file:
		importlib.import_module(file.split('.')[0])
# from .centernet_trainer import CenterNetTrainer
# from .unetpp_trainer import UnetppTrainer
# from .erfnet_trainer import erfnetTrainer
# from .efficientdet_trainer import EfficientDetTrainer
# from .SAN_trainer import SANTrainer

# _train_factory = {
# 	'centernet': CenterNetTrainer,
# 	'unetppseg': UnetppTrainer,
# 	'erfnetseg': erfnetTrainer,
# 	'efficientdet': EfficientDetTrainer,
# 	'SAN': SANTrainer,
# }

# def get_trainer(opt):
# 	return _train_factory[opt.task](opt)