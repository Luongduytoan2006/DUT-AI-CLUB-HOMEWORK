import torch
import torch.nn as nn


class ResUnit(nn.Module):
	expansion = 1

	def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=3,
			stride=stride,
			padding=1,
			bias=False,
		)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(
			out_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			padding=1,
			bias=False,
		)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		skip = self.shortcut(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = x + skip
		x = self.relu(x)
		return x


class myResNet18(nn.Module):
	def __init__(self, num_classes: int = 1000, drop_ratio: float = 0.5) -> None:
		super().__init__()
		self.current_channels = 64

		self.stem = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)

		self.layer1 = self._stack(64, 2, stride=1)
		self.layer2 = self._stack(128, 2, stride=2)
		self.layer3 = self._stack(256, 2, stride=2)
		self.layer4 = self._stack(512, 2, stride=2)

		self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.head_dropout = nn.Dropout(p=drop_ratio)
		self.head = nn.Linear(512 * ResUnit.expansion, num_classes)

	def _stack(self, out_channels: int, block_count: int, stride: int) -> nn.Sequential:
		layers = [ResUnit(self.current_channels, out_channels, stride=stride)]
		self.current_channels = out_channels * ResUnit.expansion
		for _ in range(1, block_count):
			layers.append(ResUnit(self.current_channels, out_channels, stride=1))
		return nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.global_pool(x)
		x = torch.flatten(x, 1)
		x = self.head_dropout(x)
		x = self.head(x)
		return x


if __name__ == "__main__":
	model = myResNet18(num_classes=10, drop_ratio=0.5)
	x = torch.randn(2, 3, 224, 224)
	y = model(x)
	print("myResNet18 output shape:", y.shape)
