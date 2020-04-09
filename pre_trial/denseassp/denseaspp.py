from basemodel.densenet import *

class DenseASPP(nn.Module):
	def __init__(self, nclass, pretrained_base=True, dilate_scale=8, **kwargs):
		super(DenseASPP, self).__init__()
		self.nclass = nclass
		self.dilate_scale = dilate_scale
		self.pretrained = get_dilated_densenet(121, dilate_scale, pretrained=pretrained_base)

		in_channels = self.pretrained.num_features

		self.head = _DenseASPPHead(in_channels, 64)

		#ding
		self.end_transpose = nn.ConvTranspose2d(64,32,8,stride=8)
		self.final_conv = nn.Conv2d(32,nclass,3,stride=1,padding=1)

	def forward(self, x):
		size = x.size()[2:]
		features = self.pretrained.features(x)
		if self.dilate_scale > 8:
			features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=True)

		x = self.head(features)
		# x = F.interpolate(x, size, mode='bilinear', align_corners=True)
		#ding
		x = self.end_transpose(x)
		x = self.final_conv(x)

		return x


class _DenseASPPHead(nn.Module):
	def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
		super(_DenseASPPHead, self).__init__()
		self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64, norm_layer, norm_kwargs)
		self.block = nn.Sequential(
			nn.Dropout(0.1),
			nn.Conv2d(in_channels+5*64, nclass, 1)
		)

	def forward(self, x):
		x = self.dense_aspp_block(x)
		return self.block(x)

class _DenseASPPConv(nn.Sequential):
	def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
				 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
		super(_DenseASPPConv, self).__init__()
		self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
		self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
		self.add_module('relu1', nn.ReLU(True)),
		self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
		self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
		self.add_module('relu2', nn.ReLU(True)),
		self.drop_rate = drop_rate

	def forward(self, x):
		features = super(_DenseASPPConv, self).forward(x)
		if self.drop_rate > 0:
			features = F.dropout(features, p=self.drop_rate, training=self.training)
		return features


class _DenseASPPBlock(nn.Module):
	def __init__(self, in_channels, inter_channels1, inter_channels2,
				norm_layer=nn.BatchNorm2d, norm_kwargs=None):
		super(_DenseASPPBlock, self).__init__()
		self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
									norm_layer, norm_kwargs)
		self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
									norm_layer, norm_kwargs)
		self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
									norm_layer, norm_kwargs)
		self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
									norm_layer, norm_kwargs)
		self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
									norm_layer, norm_kwargs)

	def forward(self, x):
		aspp3 = self.aspp_3(x)
		x = torch.cat([aspp3, x], dim=1)

		aspp6 = self.aspp_6(x)
		x = torch.cat([aspp6, x], dim=1)

		aspp12 = self.aspp_12(x)
		x = torch.cat([aspp12, x], dim=1)

		aspp18 = self.aspp_18(x)
		x = torch.cat([aspp18, x], dim=1)

		aspp24 = self.aspp_24(x)
		x = torch.cat([aspp24, x], dim=1)

		return x


def get_denseaspp(nclass=2, pretrained_base=True, pretrained_weight=None):
	model = DenseASPP(nclass, pretrained_base=pretrained_base)
	if pretrained_weight:
		model.load_state_dict(torch.load(pretrained_weight), strict=False)
	return model

if __name__ == '__main__':
	img = torch.randn(1,3,480,480)
	model = get_denseaspp()
	outputs = model(img)
	# print(outputs.size())