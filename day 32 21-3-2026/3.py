import torch
import torch.nn as nn


class DWPointUnit(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, step: int = 1) -> None:
		super().__init__()
		self.unit = nn.Sequential(
			nn.Conv2d(
				in_channels,
				in_channels,
				kernel_size=3,
				stride=step,
				padding=1,
				groups=in_channels,
				bias=False,
			),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.unit(x)


class myMobileNet(nn.Module):
	def __init__(self, num_classes: int = 1000, drop_ratio: float = 0.2) -> None:
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
		)

		self.unit_specs = [
			(32, 64, 1),
			(64, 128, 2),
			(128, 128, 1),
			(128, 256, 2),
			(256, 256, 1),
			(256, 512, 2),
			(512, 512, 1),
			(512, 512, 1),
			(512, 512, 1),
			(512, 512, 1),
			(512, 512, 1),
			(512, 1024, 2),
			(1024, 1024, 1),
		]
		self.body = nn.ModuleList([DWPointUnit(c_in, c_out, step=s) for c_in, c_out, s in self.unit_specs])

		self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.head_dropout = nn.Dropout(p=drop_ratio)
		self.head = nn.Linear(1024, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		for unit in self.body:
			x = unit(x)
		x = self.global_pool(x)
		x = torch.flatten(x, 1)
		x = self.head_dropout(x)
		x = self.head(x)
		return x


if __name__ == "__main__":
	model = myMobileNet(num_classes=10, drop_ratio=0.2)
	x = torch.randn(2, 3, 224, 224)
	y = model(x)
	print("myMobileNet output shape:", y.shape)
