import sys
sys.path.insert(0, './unetscnn')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from model import *
import cv2
from torchvision import transforms
import numpy as np

input_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
])
cuda = torch.device('cuda:0')

net = get_erfnet(pretrained=True, pretrained_weights='./snapshot/erfnet_curbstonedata_best_model.pth')

net.eval()
net = net.to(cuda)

frameToStart = 10000
video = cv2.VideoCapture('../_camera_image_raw2.mp4')
video.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)

success = True
while success:
	success, frame = video.read()
	width = frame.shape[1]
	height = frame.shape[0]

	img = cv2.resize(frame, (512,256))

	img = input_transform(img)
	img = torch.unsqueeze(img, 0)
	img = img.to(cuda)

	_,pred = net(img, False)
	out = torch.argmax(pred, 1)
	out = torch.squeeze(out)

	binary = out.cpu().numpy().astype('uint8')
	binary = cv2.resize(binary, (width,height))
	zero=np.zeros(binary.shape)
	aa=np.stack((zero,zero,binary*255),axis=2)
	aaa=cv2.addWeighted(frame.astype('uint8'),1,aa.astype('uint8'),200,0)

	cv2.imshow('video', aaa)
	c = cv2.waitKey(1)
	if c == 27:
		break