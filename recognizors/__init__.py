from utils.registry import get_recognizor
import os,sys,importlib
sys.path.append(os.path.dirname(__file__))
for file in os.listdir(os.path.dirname(__file__)):
	if '.py' in file and 'base' not in file and '__init__' not in file:
		importlib.import_module(file.split('.')[0])

# from .centernetdet import CenterNetdet
# from .unetppdet import Unetppdet
# from .efficientdet import Efficientdet
# from .sancls import SANcls

# _recognizor_factory = {
# 	'centernet': CenterNetdet,
# 	'unetppseg': Unetppdet,
# 	'efficientdet': Efficientdet,
# 	'SAN': SANcls,
# }

# def get_recognizor(opt):
# 	return _recognizor_factory[opt.task](opt)