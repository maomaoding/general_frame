import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os,torch,cv2
from denseaspp import *
from PIL import Image
import numpy as np
from torchvision import transforms

input_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
])

cuda0 = torch.device('cuda:0')

pretrained_weight = '/home/ubuntu/dyh/denseassp/snapshot/pre_denseaspp_curbstonedata_best_model.pth'

model = get_denseaspp(pretrained_weight=pretrained_weight)
model.eval()
model = model.to(cuda0)

# img_patha = '/home/ubuntu/dyh/curbstonedata/dyh/val_img/1557985956251l.jpg'
img_root_path = '/home/ubuntu/dyh/curbstonedata/dyh/val_img'
out_pred_path = '/home/ubuntu/dyh/denseassp/test'

for file in os.listdir(img_root_path):
	img_path = os.path.join(img_root_path, file)
	print(img_path)
	ori_img = cv2.imread(img_path)
	img = ori_img[:,:,::-1]
	width = img.shape[1]
	height = img.shape[0]

	img = cv2.resize(img, (480,480))

	img = input_transform(img)
	img = torch.unsqueeze(img, 0)
	img = img.to(cuda0)

	pred = model(img)
	out = torch.argmax(pred, 1)
	out = torch.squeeze(out)

	binary = out.cpu().numpy().astype('uint8')
	binary = cv2.resize(binary, (width,height))
	zero=np.zeros(binary.shape)
	aa=np.stack((binary*255,zero,zero),axis=2)
	aaa=cv2.addWeighted(ori_img.astype('uint8'),1,aa.astype('uint8'),200,0)
	cv2.imwrite(os.path.join(out_pred_path, file), aaa)
	# print(np.unique(binary))