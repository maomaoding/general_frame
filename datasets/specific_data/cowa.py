from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os
import torch.utils.data as data


class Cowa(data.Dataset):

    num_classes = 7

    default_resolution = [512, 720] #[h,w]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)


    def get_anns(self, index):
        label_path = self.label_path[index]
        json_labels=open(label_path,"r").readlines()[0]
        label_dict=json.loads(json_labels)
        labels = []
        for bb in label_dict["object"]:
            label = [ bb["bbox"]["x"],
                      bb["bbox"]["y"],
                      bb["bbox"]["x"]+bb["bbox"]["w"],
                      bb["bbox"]["y"]+bb["bbox"]["h"],
                      self.class_name.index(bb["category"])]

            labels.append(label)
        labels = np.array(labels)
        #print(labels)
        return labels

    def get_img_path(self, index):
        return self.img_path[index]

    def get_dataset_size(self):
        return self.num_samples

    def __init__(self, opt, split="train"):
        super(Cowa, self).__init__(opt)
        self.data_dir = os.path.join(opt.data_dir, 'cowa_obj')
        self.annot_path = os.path.join(self.data_dir,'{}.txt').format(split)
        self.class_name = ['car','bus','pedestrian','truck','cyclist',"tricycle","special"]
        self.split = split
        self.opt = opt

        print('==> initializing cowa {} data.'.format(split))
        self.img_path=[]
        self.label_path=[]
        for path in open(self.annot_path, 'r'):
            self.img_path.append(os.path.join( self.data_dir,path.rstrip().split(" ")[0]))
            self.label_path.append(os.path.join(self.data_dir,path.rstrip().split(" ")[1]))
        self.num_samples = len(self.img_path)
        print('Loaded {} {} samples'.format(split, self.num_samples))


    def __len__(self):
        return self.num_samples

