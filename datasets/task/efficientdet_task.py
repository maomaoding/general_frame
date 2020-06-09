from .base import BaseTask
import numpy as np
import torch

class efficientDetDataset(BaseTask):
	def get_data(self, img, anns):
		return {'img': img, 'annot': anns};

	def collate_fn(self, data):
		imgs = [s['img'] for s in data]
		annots = [s['annot'] for s in data]

		imgs = torch.from_numpy(np.stack(imgs, axis=0))

		max_num_annots = max(annot.shape[0] for annot in annots)

		if max_num_annots > 0:
			annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
			for idx, annot in enumerate(annots):
				if annot.shape[0] > 0:
					annot_padded[idx, :annot.shape[0], :] = torch.from_numpy(annot)
		else:
			annot_padded = torch.ones((len(annots), 1, 5)) * -1
			
		return {'img': imgs, 'annot': annot_padded}