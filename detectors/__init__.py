from .centernetdet import CenterNetdet
from .unetppdet import Unetppdet

_detector_factory = {
	'centernet': CenterNetdet,
	'unetppseg': Unetppdet,
}

def get_detector(opt):
	return _detector_factory[opt.task](opt)