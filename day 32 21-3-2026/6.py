import torch
import torch.nn as nn


class ChannelNorm2d(nn.Module):
	def __init__(self, channels: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.norm = nn.LayerNorm(channels, eps=eps)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x.permute(0, 2, 3, 1)
		x = self.norm(x)
		x = x.permute(0, 3, 1, 2)
		return x


class ConvNeXtBlock(nn.Module):
	def __init__(self, dim: int, drop_path: float = 0.0) -> None:
		super().__init__()
		self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
		self.norm = ChannelNorm2d(dim)
		self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
		self.act = nn.GELU()
		self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
		self.dropout = nn.Dropout2d(p=drop_path) if drop_path > 0 else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x
		x = self.dwconv(x)
		x = self.norm(x)
		x = self.pwconv1(x)
		x = self.act(x)
		x = self.pwconv2(x)
		x = self.dropout(x)
		return x + identity


class myConvNeXtTiny(nn.Module):
	def __init__(self, num_classes: int = 1000, drop_ratio: float = 0.5) -> None:
		super().__init__()
		dims = (96, 192, 384, 768)
		depths = (3, 3, 9, 3)

		self.stem = nn.Sequential(
			nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
			nn.BatchNorm2d(dims[0]),
			ChannelNorm2d(dims[0]),
		)

		self.stage0 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(depths[0])])
		self.down0 = nn.Sequential(ChannelNorm2d(dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))

		self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(depths[1])])
		self.down1 = nn.Sequential(ChannelNorm2d(dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))

		self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(depths[2])])
		self.down2 = nn.Sequential(ChannelNorm2d(dims[2]), nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2))

		self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(depths[3])])

		self.norm = nn.LayerNorm(dims[-1])
		self.head_dropout = nn.Dropout(p=drop_ratio)
		self.head = nn.Linear(dims[-1], num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem(x)
		x = self.stage0(x)
		x = self.down0(x)
		x = self.stage1(x)
		x = self.down1(x)
		x = self.stage2(x)
		x = self.down2(x)
		x = self.stage3(x)

		x = x.mean(dim=(2, 3))
		x = self.norm(x)
		x = self.head_dropout(x)
		x = self.head(x)
		return x


if __name__ == "__main__":
	model = myConvNeXtTiny(num_classes=10, drop_ratio=0.5)
	x = torch.randn(2, 3, 224, 224)
	y = model(x)
	print("myConvNeXtTiny output shape:", y.shape)
