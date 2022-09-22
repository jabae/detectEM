import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv3x3x3(in_channels, out_channels):

	return nn.Conv3d(in_channels, out_channels,
									kernel_size=(3,3,3), stride=1,
									padding=(1,1,1), bias=True)


def maxpool2x2():

	return nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2),
											padding=0)


def maxpool2x2x2():

	return nn.MaxPool3d(kernel_size=(2,2,2), stride=2,
											padding=0)


class UpConv2x2(nn.Module):

	def __init__(self, channels):

		super(UpConv2x2, self).__init__()
		self.upsample = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
		self.conv = nn.Conv3d(channels, channels//2, kernel_size=(1,1,1),
													stride=1, padding=0, bias=True)

	def forward(self, x):

		x = self.upsample(x)
		x = self.conv(x)

		return x


def concat(xh, xv):

	return torch.cat([xh, xv], dim=1)


# Convolution block
class ConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):
	
		super(ConvBlock3, self).__init__()
		self.conv1 = conv3x3x3(in_channels, out_channels)
		self.conv2 = conv3x3x3(out_channels, out_channels)
		self.norm = nn.BatchNorm3d(out_channels, track_running_stats=False)

	def forward(self, x):

		x = F.relu(self.norm(self.conv1(x)))
		x = F.relu(self.norm(self.conv2(x)))

		return x


# Downconvolution block
class DownConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(DownConvBlock3, self).__init__()
		self.downsample = maxpool2x2()
		self.convblock = ConvBlock3(in_channels, out_channels)

	def forward(self, x):

		x = self.downsample(x)
		x = self.convblock(x)

		return x


# Upconvolution block
class UpConvBlock3(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(UpConvBlock3, self).__init__()
		self.upsample = UpConv2x2(in_channels)
		self.convblock = ConvBlock3(in_channels//2 + out_channels, out_channels)

	def forward(self, xh, xv):

		xv = self.upsample(xv)
		x = concat(xh, xv)
		x = self.convblock(x)

		return x


class ConvOut(nn.Module):

	def __init__(self, in_channels, out_channels):

		super(ConvOut, self).__init__()
		self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1,
													padding=0, bias=True)

	def forward(self, x):

		x = self.conv(x)

		return x


# Network architecture
class UNet(nn.Module):

	def __init__(self):

		super(UNet, self).__init__()
		fs = [16,32,64,128,256]
		self.conv_in = ConvBlock3(1, fs[0])
		self.dconv1 = DownConvBlock3(fs[0], fs[1])
		self.dconv2 = DownConvBlock3(fs[1], fs[2])
		self.dconv3 = DownConvBlock3(fs[2], fs[3])
		self.dconv4 = DownConvBlock3(fs[3], fs[4])

		self.uconv1 = UpConvBlock3(fs[4], fs[3])
		self.uconv2 = UpConvBlock3(fs[3], fs[2])
		self.uconv3 = UpConvBlock3(fs[2], fs[1])
		self.uconv4 = UpConvBlock3(fs[1], fs[0])
		self.conv_out = conv3x3(fs[0], 1)

		self._initialize_weights()

	def forward(self, x):

		x1 = self.conv_in(x)
		x2 = self.dconv1(x1)
		x3 = self.dconv2(x2)
		x4 = self.dconv3(x3)
		x5 = self.dconv4(x4)
		x6 = self.uconv1(x4, x5)
		x7 = self.uconv2(x3, x6)
		x8 = self.uconv3(x2, x7)
		x9 = self.uconv4(x1, x8)
		x10 = self.conv_out(x9)

		return x10

	def _initialize_weights(self):

		conv_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
		for m in conv_modules:

			n = m.weight.shape[1]*m.weight.shape[2]*m.weight.shape[3]*m.weight.shape[4]
			m.weight.data.normal_(0, np.sqrt(2./n))
			m.bias.data.zero_()
