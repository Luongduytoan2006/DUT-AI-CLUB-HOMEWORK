import torch
import torch.nn as nn


class AlexConvUnit(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int,
		stride: int = 1,
		padding: int = 0,
	) -> None:
		super().__init__()
		self.unit = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.unit(x)


class myAlexNet(nn.Module):
	def __init__(self, num_classes: int = 1000, drop_ratio: float = 0.5) -> None:
		super().__init__()
		self.stage1 = nn.Sequential(
			AlexConvUnit(3, 64, kernel_size=11, stride=4, padding=2),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.stage2 = nn.Sequential(
			AlexConvUnit(64, 192, kernel_size=5, padding=2),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.stage3 = nn.Sequential(
			AlexConvUnit(192, 384, kernel_size=3, padding=1),
			AlexConvUnit(384, 256, kernel_size=3, padding=1),
			AlexConvUnit(256, 256, kernel_size=3, padding=1),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)

		self.spatial_pool = nn.AdaptiveAvgPool2d((6, 6))
		self.head = nn.Sequential(
			nn.Dropout(p=drop_ratio),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=drop_ratio),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.spatial_pool(x)
		x = torch.flatten(x, 1)
		x = self.head(x)
		return x


if __name__ == "__main__":
	model = myAlexNet(num_classes=10, drop_ratio=0.5)
	x = torch.randn(4, 3, 224, 224)
	y = model(x)
	print("myAlexNet output shape:", y.shape)