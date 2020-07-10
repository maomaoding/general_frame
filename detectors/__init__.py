from .centernetdet import CenterNetdet
from .unetppdet import Unetppdet
from .efficientdet import Efficientdet

_detector_factory = {
	'centernet': CenterNetdet,
	'unetppseg': Unetppdet,
	'efficientdet': Efficientdet,
}

def get_detector(opt):
	return _detector_factory[opt.task](opt)