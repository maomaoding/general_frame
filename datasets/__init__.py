from .specific_data.coco import COCO
from .specific_data.bdd import BDD
from .specific_data.TT100k import TT100k
from .specific_data.junctions import Junctions
from .specific_data.lane import CowaLane, Culane
from .specific_data.furnitures import Furnitures
from .task.centernet_task import CTDetDataset
from .task.seg_task import SegmentDataset
from .task.efficientdet_task import efficientDetDataset
from .task.SAN_task import SANDataset
from .task.detr_task import detrDataset

_dataset_factory = {
	'coco': COCO,
	'bdd100k': BDD,
	'TT100K': TT100k,
	'lane': CowaLane,
	'CULane': Culane,
	'junctions': Junctions,
	'furnitures': Furnitures,
}

_task_factory = {
	'centernet': CTDetDataset,
	'unetppseg': SegmentDataset,
	'erfnetseg': SegmentDataset,
	'efficientdet': efficientDetDataset,
	'SAN': SANDataset,
	'detr': detrDataset,
}

def get_dataset(opt, split='train'):
	class Dataset(_task_factory[opt.task], _dataset_factory[opt.dataset]):
		pass
	return Dataset(opt, split)