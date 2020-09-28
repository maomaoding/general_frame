#coding:utf8
import torch as t
from torch import nn
import torch.nn.functional as F
from .base_models.resnet50 import RESNET
import os
from utils.registry import *

decoder_filters = [64, 128, 256, 512, 1024]
class BN_CONV_BN(nn.Module):
	def __init__(self,input_shape,output_shape):
		super(BN_CONV_BN, self).__init__()
		self.bn_in = t.nn.BatchNorm2d(input_shape)
		self.conv=t.nn.Conv2d(input_shape,output_shape,[3,3],1,1)
		self.bn=t.nn.BatchNorm2d(output_shape)
		self.relu=nn.ReLU()

	def forward(self, x):
		x=self.bn_in(x)
		x = self.relu(x)
		x=self.conv(x)
		x=self.bn(x)
		x=self.relu(x)
		return x

#added by dingyaohua
class Lane_classifier(nn.Module):
	def __init__(self, num_labels=4):
		super(Lane_classifier, self).__init__()

		self.num_labels = num_labels

		if self.num_labels == 4:
			self.layers = nn.ModuleList()

			self.layers.append(nn.Conv2d(2048, 32, (3, 3), stride=1, padding=(4,4), bias=False, dilation=(4,4)))
			self.layers.append(nn.BatchNorm2d(32, eps=1e-03))
			self.layers.append(nn.ReLU())
			self.layers.append(nn.Dropout2d(0.1))
			self.layers.append(nn.Conv2d(32, 5, (1,1), stride=1, padding=(0,0), bias=True))

			self.maxpool = nn.MaxPool2d(2, stride=2)
			self.linear11 = nn.Linear(160, 128)
			self.linear22 = nn.Linear(128, num_labels*5) #4 lane 5 labels
		elif self.num_labels == 7:
			self.conv_=t.nn.Conv2d(2048,8,[1,1],1,0)
			self.bn_=t.nn.BatchNorm2d(8)
			self.softmax=t.nn.Softmax(dim=1)
			self.pooling_ = t.nn.AvgPool2d((2,2),(2,2))
			self.fc9_ = nn.Linear(8 * 4*8, 128)
			self.fc10_ = nn.Linear(128, 7)
			self.sigmoid=nn.Sigmoid()

	def forward(self, input):
		output = input

		if self.num_labels == 4:
			for layer in self.layers:
				output = layer(output)

			output = F.softmax(output, dim=1)
			output = self.maxpool(output)
			output = output.view(output.size()[0], -1)
			# output = output.view(1, -1)
			output = self.linear11(output)
			output = F.relu(output)
			output = self.linear22(output)
			output = output.view(output.size()[0], 5, 4)
			# output = output.view(1, 5, 4)
		elif self.num_labels == 7:
			output = self.conv_(output)
			output = self.bn_(output)

			output = self.softmax(output)
			output = self.pooling_(output)
			output = output.view(-1, 256)
			output = self.fc9_(output)
			output = F.relu(output)
			output = self.fc10_(output)
			output = self.sigmoid(output)

		return output

class UNETPP(nn.Module):
	def __init__(self, num_classes=8, num_labels=4, export_onnx=False):
		super(UNETPP, self).__init__()
		self.export_onnx = export_onnx
		self.num_labels = num_labels
		self.model_name = "UNETPP"
		self.resnet = RESNET()
		self.deconv_up12=t.nn.ConvTranspose2d(256,64,[2,2],2)
		self.convbn_up12=BN_CONV_BN(128,64)

		self.deconv_up22 = t.nn.ConvTranspose2d(512, 256, [2, 2], 2)
		self.convbn_up22 = BN_CONV_BN(512, 256)
		self.deconv_up13 = t.nn.ConvTranspose2d(256, 128, [2, 2], 2)
		self.convbn_up13 = BN_CONV_BN(256, 64)

		self.deconv_up32 = t.nn.ConvTranspose2d(1024, 512, [2, 2], 2)
		self.convbn_up32 = BN_CONV_BN(1024, 512)
		self.deconv_up23 = t.nn.ConvTranspose2d(512, 256, [2, 2], 2)
		self.convbn_up23 = BN_CONV_BN(768, 128)
		self.deconv_up14 = t.nn.ConvTranspose2d(128, 64, [2, 2], 2)
		self.convbn_up14 = BN_CONV_BN(256, 64)

		self.deconv_up42 = t.nn.ConvTranspose2d(2048, 1024, [2, 2], 2)
		self.convbn_up42 = BN_CONV_BN(2048, 1024)
		self.deconv_up33 = t.nn.ConvTranspose2d(1024, 512, [2, 2], 2)
		self.convbn_up33 = BN_CONV_BN(1536, 256)
		self.deconv_up24 = t.nn.ConvTranspose2d(256, 128, [2, 2], 2)
		self.convbn_up24 = BN_CONV_BN(768, 128)
		self.deconv_up15 = t.nn.ConvTranspose2d(128, 64, [2, 2], 2)
		self.convbn_up15 = BN_CONV_BN(320, 64)

		self.deconv_final_12_= t.nn.ConvTranspose2d(64, num_classes, [2, 2], 2)
		self.deconv_final_13_= t.nn.ConvTranspose2d(64, num_classes, [2, 2], 2)
		self.deconv_final_14_ = t.nn.ConvTranspose2d(64, num_classes, [2, 2], 2)
		self.deconv_final_15_ = t.nn.ConvTranspose2d(64, num_classes, [2, 2], 2)
		self.bn_12_ = t.nn.BatchNorm2d(num_classes)
		self.bn_13_ = t.nn.BatchNorm2d(num_classes)
		self.bn_14_ = t.nn.BatchNorm2d(num_classes)
		self.bn_15_ = t.nn.BatchNorm2d(num_classes)

		# self.conv_=t.nn.Conv2d(2048,8,[1,1],1,0)
		# self.bn_=t.nn.BatchNorm2d(8)
		self.softmax=t.nn.Softmax(dim=1)
		# self.pooling_ = t.nn.AvgPool2d((2,2),(2,2))
		# self.fc9_ = nn.Linear(8 * 4*8, 128)
		# self.fc10_ = nn.Linear(128, 5)
		# self.sigmoid=nn.Sigmoid()

		#added by dingyaohua
		self.lane_classifier = Lane_classifier(num_labels)


	def forward(self, x):
		n_upsample_blocks=4
		stage4_unit1_relu1,stage3_unit1_relu1,stage2_unit1_relu1,relu0,\
		relu1, stage3_unit2_relu1, stage2_unit2_relu1, stage1_unit2_relu1=self.resnet(x)
		#added by dingyaohua
		lane_classes = self.lane_classifier(relu1)

		# x2 = self.conv_(relu1)
		# x2=self.bn_(x2)

		# x2 =self.softmax(x2)
		# x2=self.pooling_(x2)
		# x2=x2.view(-1, 256)
		# x2 = self.fc9_(x2)
		# x2 = F.relu(x2)
		# x2 = self.fc10_(x2)
		# x2=self.sigmoid(x2)
		up1_2=self.deconv_up12(stage1_unit2_relu1)
		conv1_2=t.cat([relu0,up1_2],1)
		conv1_2=self.convbn_up12(conv1_2)

		up2_2 = self.deconv_up22(stage2_unit2_relu1)
		conv2_2 = t.cat([stage2_unit1_relu1, up2_2],1)
		conv2_2 = self.convbn_up22(conv2_2)
		up1_3 = self.deconv_up13(conv2_2)
		conv1_3 = t.cat([up1_3,relu0,conv1_2],1)
		conv1_3 = self.convbn_up13(conv1_3)

		up3_2 = self.deconv_up32(stage3_unit2_relu1)
		conv3_2 =t.cat([up3_2, stage3_unit1_relu1],1)
		conv3_2 = self.convbn_up32(conv3_2)
		up2_3 = self.deconv_up23(conv3_2)
		conv2_3 = t.cat([up2_3, stage2_unit1_relu1, conv2_2],1)
		conv2_3 = self.convbn_up23(conv2_3)
		up1_4 = self.deconv_up14(conv2_3)
		conv1_4 = t.cat([up1_4, relu0, conv1_2, conv1_3],1)
		conv1_4 = self.convbn_up14(conv1_4)

		up4_2 =  self.deconv_up42(relu1)
		conv4_2 = t.cat([up4_2, stage4_unit1_relu1],1)
		conv4_2 = self.convbn_up42(conv4_2)
		up3_3 = self.deconv_up33(conv4_2)
		conv3_3 = t.cat([up3_3, stage3_unit1_relu1, conv3_2], 1)
		conv3_3 = self.convbn_up33(conv3_3)

		up2_4 = self.deconv_up24(conv3_3)
		conv2_4 = t.cat([up2_4, stage2_unit1_relu1, conv2_2, conv2_3], 1)
		conv2_4 = self.convbn_up24(conv2_4)
		up1_5 = self.deconv_up15(conv2_4)
		conv1_5 = t.cat([up1_5, relu0, conv1_2, conv1_3, conv1_4],1)
		conv1_5 = self.convbn_up15(conv1_5)

		conv1_2 = self.deconv_final_12_(conv1_2)
		conv1_3 = self.deconv_final_13_(conv1_3)
		conv1_4 = self.deconv_final_14_(conv1_4)
		conv1_5 = self.deconv_final_15_(conv1_5)
		conv1_2 = self.bn_12_(conv1_2)
		conv1_3 = self.bn_13_(conv1_3)
		conv1_4 = self.bn_14_(conv1_4)
		conv1_5 = self.bn_15_(conv1_5)
		if not self.export_onnx:
			ret = {'spatial': [conv1_2,conv1_3,conv1_4,conv1_5],
					'label': lane_classes}
		else:
			conv1_2=self.softmax(conv1_2)
			conv1_3 = self.softmax(conv1_3)
			conv1_4 = self.softmax(conv1_4)
			conv1_5 = self.softmax(conv1_5)
			if self.num_labels == 4:
				lane_classes = self.softmax(lane_classes)
				lane_classes = lane_classes.view(1,5,4,1)

			ret = [conv1_5,lane_classes]

		return ret

	''' def forward(self, x):
		x = self.resnet(x)
		x= self.udlr(x)
		x = self.conv8(x)
		x1 = F.interpolate(x, size=[288, 800], mode='bilinear', align_corners=False)
		# x1 = F.softmax(x1, dim=1)
		x2 = F.softmax(x, dim=1)
		x2 = F.avg_pool2d(x2, 2, stride=2, padding=0)
		x2 = x2.view(-1, x2.numel() // x2.shape[0])
		x2 = self.fc9(x2)
		x2 = F.relu(x2)
		x2 = self.fc10(x2)
		x2 = t.sigmoid(x2)

		return x1, x2'''

@register_model
def unetpp_model(opt):
	return UNETPP(opt.num_classes, opt.num_labels, opt.export_onnx)

if __name__ == '__main__':
	input = t.randn(5,3,256,512)
	net = UNETPP()
	output = net(input)
	print(output['label'].size())
