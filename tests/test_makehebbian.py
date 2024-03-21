import torch
import torch.nn as nn
from hebb.makehebbian import makehebbian, UnsqueezeLast, FlattenLast

def test_makehebbian():
	class Net(nn.Module):
		def __init__(self):
			super().__init__()
			self.down = nn.Sequential(
				nn.Conv2d(3, 16, 3, stride=2),
				nn.BatchNorm2d(16),
				nn.ReLU(),
			)
			self.up = nn.Sequential(
				nn.ConvTranspose2d(16, 20, 3, stride=2),
				nn.BatchNorm2d(20),
				nn.ReLU(),
			)
			
			self.clf = nn.Sequential(
				FlattenLast(2),
				nn.Linear(20, 16),
				nn.BatchNorm1d(16),
				nn.ReLU(),
				nn.Dropout(0.5),
				nn.Linear(16, 10),
			)
		
		def forward(self, x):
			return self.clf(self.up(self.down(x)))
		
	net = Net()
	print(net)
	print(*[n  for n, _ in net.named_modules()])
	print(*[n for n, _ in net.named_parameters()])
	makehebbian(net, exclude=['clf.5'], hebb_params={})
	print(net)
	print(*[n for n, _ in net.named_modules()])
	print(*[n for n, _ in net.named_parameters()])

def test_makehebbian3d():
	from models.getnetwork import get_network
	net = get_network('unet3d', 3, 2)
	
	makehebbian(net, exclude=['conv'], hebb_params={})
	
	print(net)
	
	net.train()
	inputs = torch.randn(4, 3, 16, 128, 128)
	outputs = net(inputs)
	print(outputs.shape)

if __name__ == '__main__':
	test_makehebbian3d()