import torch,time,os
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock (nn.Module):
	def __init__(self, ninput, noutput):
		super().__init__()

		self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
		self.pool = nn.MaxPool2d(2, stride=2)
		self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

	def forward(self, input):
		output = torch.cat([self.conv(input), self.pool(input)], 1)
		output = self.bn(output)
		return F.relu(output)
	

class non_bottleneck_1d (nn.Module):
	def __init__(self, chann, dropprob, dilated):        
		super().__init__()

		self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

		self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

		self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

		self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

		self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

		self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

		self.dropout = nn.Dropout2d(dropprob)
		

	def forward(self, input):

		output = self.conv3x1_1(input)
		output = F.relu(output)
		output = self.conv1x3_1(output)
		output = self.bn1(output)
		output = F.relu(output)

		output = self.conv3x1_2(output)
		output = F.relu(output)
		output = self.conv1x3_2(output)
		output = self.bn2(output)

		if (self.dropout.p != 0):
			output = self.dropout(output)
		
		return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
	def __init__(self, in_channels, num_classes):
		super().__init__()
		self.initial_block = nn.Sequential(
								nn.Conv2d(in_channels, 16, (3,3), stride=1, padding=1),
								nn.BatchNorm2d(16),
								nn.ReLU(),
								non_bottleneck_1d(16, 0, 1),
								nn.Conv2d(16, 8, (1,1), stride=1),
								nn.BatchNorm2d(8),
								nn.ReLU(),
							)
		self.initial_downsampler = DownsamplerBlock(8, 16)

		self.layers_d1 = nn.Sequential(
							non_bottleneck_1d(16, 0, 1),
							non_bottleneck_1d(16, 0, 1),
							nn.Conv2d(16, 32, (1,1), stride=1),
							nn.BatchNorm2d(32),
							nn.ReLU(),
							DownsamplerBlock(32,64),
						)

		self.layers_d2 = nn.Sequential(
							non_bottleneck_1d(64, 0, 2),
							non_bottleneck_1d(64, 0, 2),
							nn.Conv2d(64, 64, (1,1), stride=1),
							nn.BatchNorm2d(64),
							nn.ReLU(),
							DownsamplerBlock(64,128),
						)

		self.layers = nn.ModuleList()

		for x in range(0, 1):    #2 times
			self.layers.append(non_bottleneck_1d(128, 0.1, 1))
			self.layers.append(non_bottleneck_1d(128, 0.1, 2))
			self.layers.append(non_bottleneck_1d(128, 0.1, 4))
			self.layers.append(non_bottleneck_1d(128, 0.1, 8))

	def forward(self, input):
		output_init = self.initial_block(input)

		output_d0 = self.initial_downsampler(output_init)

		output_d1 = self.layers_d1(output_d0)

		output = self.layers_d2(output_d1)

		for layer in self.layers:
			output = layer(output)

		return output_init, output_d0, output_d1, output


class UpsamplerBlock (nn.Module):
	def __init__(self, ninput, noutput):
		super().__init__()
		self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
		self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

	def forward(self, input):
		output = self.conv(input)
		output = self.bn(output)
		return F.relu(output)

class Decoder (nn.Module):
	def __init__(self, num_classes):
		super().__init__()

		self.upblock1 = UpsamplerBlock(128,64)
		self.layers_u1 = nn.Sequential(
							nn.Conv2d(128, 64, (1,1)),
							nn.BatchNorm2d(64),
							nn.ReLU(),
							non_bottleneck_1d(64, 0, 1),
							non_bottleneck_1d(64, 0, 1),
						)

		self.upblock0 = UpsamplerBlock(64,16)
		self.layers_u0 = nn.Sequential(
							nn.Conv2d(32, 16, (1,1)),
							nn.BatchNorm2d(16),
							nn.ReLU(),
							non_bottleneck_1d(16, 0, 1),
							non_bottleneck_1d(16, 0, 1),
						)

		self.upblock = UpsamplerBlock(16,8)
		self.layers_u = nn.Sequential(
							nn.Conv2d(16, 16, (1,1)),
							nn.BatchNorm2d(16),
							nn.ReLU(),
							non_bottleneck_1d(16, 0, 1),
							non_bottleneck_1d(16, 0, 1),
						)

		self.output_conv = nn.Conv2d(16, num_classes, (1, 1))

	def forward(self, input_init, input_d0, input_d1, input):

		output_u1 = self.layers_u1(torch.cat([self.upblock1(input), input_d1], 1))
		output_u0 = self.layers_u0(torch.cat([self.upblock0(output_u1), input_d0], 1))
		output_u = self.layers_u(torch.cat([self.upblock(output_u0), input_init], 1))

		output = self.output_conv(output_u)
		return output

# ERFNet
class Net(nn.Module):
	def __init__(self, in_channels=1, out_channels=1):  #use encoder to pass pretrained encoder
		super().__init__()
		self.encoder = Encoder(in_channels, out_channels)
		self.decoder = Decoder(out_channels)
		
	def forward(self, input):
		output_init, output_d0, output_d1, output = self.encoder(input)
		decoder_output = self.decoder(output_init, output_d0, output_d1, output)
		return decoder_output

def get_erfnet(pretrained=False, pretrained_weights=None):
	net = Net(in_channels=3, out_channels=5)
	if pretrained and pretrained_weights != None:
		net.load_state_dict(torch.load(pretrained_weights), strict=False)
	return net

if __name__ == '__main__':
	net = get_erfnet()
	input = torch.randn(5,3,256,512)
	start = time.time()
	out = net(input)
	end = time.time()
	print(out.size())
	print(end-start)