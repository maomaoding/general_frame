from .networks.dla import get_dla
from .networks.unetpp import get_unetpp
from .networks.erfnet import get_erfnet
from .networks.efficientdet import get_efficientdet
_model_factory = {
	'dla34' : get_dla,
	'unetpp' : get_unetpp,
	'erfnet' : get_erfnet,
	'efficientdet': get_efficientdet,
}

def create_model(opt):
	model = _model_factory[opt.arch](opt)
	return model