from .centernetdet import CenterNetdet
from .unetppdet import Unetppdet
from .efficientdet import Efficientdet
from .sancls import SANcls

_recognizor_factory = {
	'centernet': CenterNetdet,
	'unetppseg': Unetppdet,
	'efficientdet': Efficientdet,
	'SAN': SANcls,
}

def get_recognizor(opt):
	return _recognizor_factory[opt.task](opt)