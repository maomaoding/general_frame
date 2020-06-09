import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class UnetConvBlock(nn.Module):
	def __init__(self, in_size, out_size, is_batchnorm, num_layers=2):
		super(UnetConvBlock, self).__init__()

		self.convs = nn.ModuleList()
		if is_batchnorm:
			conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
								nn.BatchNorm2d(out_size),
								nn.ReLU())
			self.convs.append(conv)
			for i in range(1, num_layers):
				conv = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding=1),
									nn.BatchNorm2d(out_size),
									nn.ReLU())
				self.convs.append(conv)
		else:
			conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
								nn.ReLU())
			self.convs.append(conv)
			for i in range(1, num_layers):
				conv = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, padding=1),
									nn.ReLU())
				self.convs.append(conv)

	def forward(self, inputs):
		outputs = inputs
		for conv in self.convs:
			outputs = conv(outputs)
		return outputs

class UnetUp(nn.Module):
	def __init__(self, in_size, out_size, is_batchnorm=False):
		super(UnetUp, self).__init__()

		self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2, groups=in_size)
		self.conv = UnetConvBlock(in_size + out_size, out_size, is_batchnorm=is_batchnorm, num_layers=2)

	def forward(self, residual, previous):
		upsampled = self.up(previous)
		result = self.conv(torch.cat([residual, upsampled], 1))
		return result

class PAM_Module(nn.Module):
	def __init__(self, in_dim):
		super(PAM_Module, self).__init__()

		self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		m_batchsize, C, height, width = x.size()
		proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
		proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

		energy = torch.bmm(proj_query, proj_key)
		attention = self.softmax(energy)
		proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

		out = torch.bmm(proj_value, attention.permute(0, 2, 1))
		out = out.view(m_batchsize, C, height, width)

		out = self.gamma*out + x
		return out

class CAM_Module(nn.Module):
	""" Channel attention module"""
	def __init__(self, in_dim):
		super(CAM_Module, self).__init__()

		self.gamma = nn.Parameter(torch.zeros(1))
		self.softmax  = nn.Softmax(dim=-1)

	def forward(self,x):
		m_batchsize, C, height, width = x.size()
		proj_query = x.view(m_batchsize, C, -1)
		proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
	   
		energy = torch.bmm(proj_query, proj_key)
		energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
		attention = self.softmax(energy_new)
		proj_value = x.view(m_batchsize, C, -1)

		out = torch.bmm(attention, proj_value)
		out = out.view(m_batchsize, C, height, width)

		out = self.gamma*out + x
		return out

################	unet with scnn	################
class unet(nn.Module):

	def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
				 groups=1, width_per_group=64, replace_stride_with_dilation=None,
				 norm_layer=None):
		super(unet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
									   dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
									   dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
									   dilate=replace_stride_with_dilation[2])

		self.up4 = UnetUp(512, 256, is_batchnorm=True)
		self.up3 = UnetUp(256, 128, is_batchnorm=True)
		self.up2 = UnetUp(128, 64, is_batchnorm=True)

		self.conv_d = nn.Conv2d(64, 64, (1, 5), padding=(0, 2), bias=False)
		self.conv_u = nn.Conv2d(64, 64, (1, 5), padding=(0, 2), bias=False)
		self.conv_r = nn.Conv2d(64, 64, (5, 1), padding=(2, 0), bias=False)
		self.conv_l = nn.Conv2d(64, 64, (5, 1), padding=(2, 0), bias=False)

		self.group_transpose = nn.ConvTranspose2d(64,64,4,stride=4,groups=64)
		self.group_1x1conv = nn.Conv2d(64,num_classes,1)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		print(x.size())
		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)

		up_4 = self.up4(x3, x4)
		up_3 = self.up3(x2, up_4)
		up_2 = self.up2(x1, up_3)

		x_split = list(torch.split(up_2, 1, 2))
		for i in range(1,len(x_split)):
			x_split[i] = x_split[i] + F.relu(self.conv_d(x_split[i-1]))
		for i in range(len(x_split)-2, 0, -1):
			x_split[i] = x_split[i] + F.relu(self.conv_u(x_split[i+1]))
		up_2 = torch.cat(x_split, dim=2)

		x_split = list(torch.split(up_2, 1, 3))
		for i in range(1,len(x_split)):
			x_split[i] = x_split[i] + F.relu(self.conv_r(x_split[i-1]))
		for i in range(len(x_split)-2, 0, -1):
			x_split[i] = x_split[i] + F.relu(self.conv_l(x_split[i+1]))
		up_2 = torch.cat(x_split, dim=3)

		x = self.group_transpose(up_2)
		x = self.group_1x1conv(x)

		return x

################	unet with attention	################
class unet_attention(nn.Module):
	def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
				 groups=1, width_per_group=64, replace_stride_with_dilation=None,
				 norm_layer=None):
		super(unet_attention, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
									   dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
									   dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
									   dilate=replace_stride_with_dilation[2])

		self.up4 = UnetUp(512, 256, is_batchnorm=True)
		self.up3 = UnetUp(256, 128, is_batchnorm=True)
		self.up2 = UnetUp(128, 64, is_batchnorm=True)

		self.pam_up4 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
			PAM_Module(256),
			nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
			)

		self.cam_up4 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
			CAM_Module(256),
			nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
			)
		self.conv_up4 = nn.Conv2d(256,256,1)

		self.pam_up3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			PAM_Module(128),
			nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
			)

		self.cam_up3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			CAM_Module(128),
			nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
			)
		self.conv_up3 = nn.Conv2d(128,128,1)

		self.pam_up2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
			PAM_Module(64),
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
			)

		self.cam_up2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
			CAM_Module(64),
			nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
			)
		self.conv_up2 = nn.Conv2d(64,64,1)

		self.predict4 = nn.Sequential(
			nn.ConvTranspose2d(256,64,16,stride=16,groups=64),
			nn.Conv2d(64,num_classes,1)
			)

		self.predict3 = nn.Sequential(
			nn.ConvTranspose2d(128,64,8,stride=8,groups=64),
			nn.Conv2d(64,num_classes,1)
			)

		self.predict2 = nn.Sequential(
			nn.ConvTranspose2d(64,64,4,stride=4,groups=64),
			nn.Conv2d(64,num_classes,1)
			)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)

		up_4 = self.up4(x3, x4)
		up_4_pam = self.pam_up4(up_4)
		up_4_cam = self.cam_up4(up_4)
		up_4 = self.conv_up4((up_4_pam+up_4_cam)*up_4)

		up_3 = self.up3(x2, up_4)
		up_3_pam = self.pam_up3(up_3)
		up_3_cam = self.cam_up3(up_3)
		up_3 = self.conv_up3((up_3_pam+up_3_cam)*up_3)

		up_2 = self.up2(x1, up_3)
		up_2_pam = self.pam_up2(up_2)
		up_2_cam = self.cam_up2(up_2)
		up_2 = self.conv_up2((up_2_pam+up_2_cam)*up_2)

		pred4 = self.predict4(up_4)
		pred3 = self.predict3(up_3)
		pred2 = self.predict2(up_2)

		return ((pred4 + pred3 + pred2) / 4)





def get_unet(pretrained_weights=None):
	model = unet(Bottleneck, [2,2,2,2])

	if pretrained_weights:
		model.load_state_dict(torch.load(pretrained_weights), strict=False)
	return model

def get_unet_attention(pretrained_weights=None):
	model = unet_attention(Bottleneck, [2,2,2,2])

	if pretrained_weights:
		model.load_state_dict(torch.load(pretrained_weights), strict=False)
	return model

if __name__ == '__main__':
	input = torch.randn(1,3,256,512)
	net = get_unet_attention()
	output = net(input)
	print(output.size())