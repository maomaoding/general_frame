from .specific_data.coco import COCO
from .specific_data.bdd import BDD
from .specific_data.TT100k import TT100k
from .specific_data.lane import CowaLane, Culane
from .task.centernet_task import CTDetDataset
from .task.seg_task import SegmentDataset

_dataset_factory = {
	'coco': COCO,
	'bdd100k': BDD,
	'TT100K': TT100k,
	'lane': CowaLane,
	'CULane': Culane,
}

_task_factory = {
	'centernet': CTDetDataset,
	'unetppseg': SegmentDataset,
	'erfnetseg': SegmentDataset,
}

def get_dataset(opt, split='train'):
	class Dataset(_dataset_factory[opt.dataset], _task_factory[opt.task]):
		pass
	return Dataset(opt, split)