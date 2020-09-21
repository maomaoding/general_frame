from .centernetdet import CenterNetdet
from .unetppdet import Unetppdet
from .efficientdet import Efficientdet
from .sancls import SANcls

_detector_factory = {
	'centernet': CenterNetdet,
	'unetppseg': Unetppdet,
	'efficientdet': Efficientdet,
	'SAN': SANcls,
}

def get_det_cls_ors(opt):
	return _detector_factory[opt.task](opt)