from .base import BaseTask
import cv2,torch
import numpy as np

class SegmentDataset(BaseTask):
	def get_data(self, img, anns):
		ret = {'input': torch.from_numpy(img), 
				'instance': torch.from_numpy(anns[0]).long(),
				# 'label': torch.from_numpy(np.array(anns[1])).long()}
				'label': torch.from_numpy(np.array(anns[1]).astype(np.float32))}
		return ret