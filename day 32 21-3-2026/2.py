import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layer(x)


class VGGStage(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, repeats: int) -> None:
		super().__init__()
		blocks: list[nn.Module] = []
		for i in range(repeats):
			c_in = in_channels if i == 0 else out_channels
			blocks.append(ConvBNReLU(c_in, out_channels))
		self.convs = nn.ModuleList(blocks)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for conv in self.convs:
			x = conv(x)
		x = self.pool(x)
		return x


class myVGG(nn.Module):
	def __init__(self, num_classes: int = 1000, drop_ratio: float = 0.5) -> None:
		super().__init__()
		self.stage1 = VGGStage(3, 64, repeats=2)
		self.stage2 = VGGStage(64, 128, repeats=2)
		self.stage3 = VGGStage(128, 256, repeats=3)
		self.stage4 = VGGStage(256, 512, repeats=3)
		self.stage5 = VGGStage(512, 512, repeats=3)

		self.spatial_pool = nn.AdaptiveAvgPool2d((7, 7))
		self.head = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=drop_ratio),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(p=drop_ratio),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stage1(x)
		x = self.stage2(x)
		x = self.stage3(x)
		x = self.stage4(x)
		x = self.stage5(x)
		x = self.spatial_pool(x)
		x = torch.flatten(x, 1)
		x = self.head(x)
		return x


if __name__ == "__main__":
	model = myVGG(num_classes=10, drop_ratio=0.5)
	x = torch.randn(2, 3, 224, 224)
	y = model(x)
	print("myVGG output shape:", y.shape)
