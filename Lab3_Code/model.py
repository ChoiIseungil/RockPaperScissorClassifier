from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_Network(nn.Module):
	def __init__(self):
		#super() function makes class inheritance more manageable and extensible
		super(Custom_Network, self).__init__()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=60, kernel_size=5, stride=1)
		self.conv2 = nn.Conv2d(in_channels=60, out_channels=200, kernel_size=4, stride=2, padding=1, padding_mode='zeros')
		self.conv3 = nn.Conv2d(in_channels=200, out_channels=400, kernel_size=2, stride=2)
		self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)
		self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1)
		self.linear8 = nn.Linear(400,1000)
		self.dropout = nn.Dropout(0.25)


	def forward(self, x):
		x = torch.relu(self.conv1(x)) #3*64*64 -> 60*60*60
		x = self.dropout(x)
		x = self.maxpool1(x) #60*60*60 -> 60*30*30
		x = self.dropout(x)
		x = torch.relu(self.conv2(x)) #60*30*30 -> 200*15*15
		x = self.dropout(x)
		x = self.maxpool2(x) # 200*15*15 -> 200*12*12
		x = self.dropout(x)
		x = torch.relu(self.conv3(x)) #400*12*12 -> 400*6*6
		x = self.dropout(x)
		x = self.avgpool(x) #400*6*6 -> 400*1*1
		x = self.dropout(x)
		x = torch.flatten(x,1)  #400*1*1 -> 400
		x = self.dropout(x)
		x= torch.relu(self.linear8(x)) #400 -> 1000
		x = self.dropout(x)
		output = F.log_softmax(x, dim = 1)  #1000 - >1
		return output