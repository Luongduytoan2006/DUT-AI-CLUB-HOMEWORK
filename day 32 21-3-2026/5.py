import torch
import torch.nn as nn


class ConvStemUnit(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int,
		stride: int = 1,
		padding: int = 0,
	) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
				bias=False,
			),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class Inception(nn.Module):
	def __init__(
		self,
		in_channels: int,
		c1: int,
		c3r: int,
		c3: int,
		c5r: int,
		c5: int,
		pool_proj: int,
	) -> None:
		super().__init__()
		self.branch1 = ConvStemUnit(in_channels, c1, kernel_size=1)

		self.branch2 = nn.Sequential(
			ConvStemUnit(in_channels, c3r, kernel_size=1),
			ConvStemUnit(c3r, c3, kernel_size=3, padding=1),
		)

		self.branch3 = nn.Sequential(
			ConvStemUnit(in_channels, c5r, kernel_size=1),
			ConvStemUnit(c5r, c5, kernel_size=5, padding=2),
		)

		self.branch4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			ConvStemUnit(in_channels, pool_proj, kernel_size=1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b1 = self.branch1(x)
		b2 = self.branch2(x)
		b3 = self.branch3(x)
		b4 = self.branch4(x)
		return torch.cat([b1, b2, b3, b4], dim=1)


class AuxHead(nn.Module):
	def __init__(self, in_channels: int, num_classes: int, drop_ratio: float = 0.7) -> None:
		super().__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
		self.proj = ConvStemUnit(in_channels, 128, kernel_size=1)
		self.fc1 = nn.Linear(128 * 4 * 4, 1024)
		self.relu = nn.ReLU(inplace=True)
		self.drop = nn.Dropout(p=drop_ratio)
		self.fc2 = nn.Linear(1024, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.avg_pool(x)
		x = self.proj(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.drop(x)
		x = self.fc2(x)
		return x


class myGoogLeNet(nn.Module):
	def __init__(self, num_classes: int = 1000, use_aux: bool = True, drop_ratio: float = 0.4) -> None:
		super().__init__()
		self.use_aux = use_aux

		self.pre_stem = nn.Sequential(
			ConvStemUnit(3, 64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool2d(3, stride=2, ceil_mode=True),
			ConvStemUnit(64, 64, kernel_size=1),
			ConvStemUnit(64, 192, kernel_size=3, padding=1),
			nn.MaxPool2d(3, stride=2, ceil_mode=True),
		)

		self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
		self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
		self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

		self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
		self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
		self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
		self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
		self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

		self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
		self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

		if use_aux:
			self.aux_head1 = AuxHead(512, num_classes)
			self.aux_head2 = AuxHead(528, num_classes)
		else:
			self.aux_head1 = None
			self.aux_head2 = None

		self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.head_drop = nn.Dropout(p=drop_ratio)
		self.head = nn.Linear(1024, num_classes)

	def forward(self, x: torch.Tensor):
		x = self.pre_stem(x)

		x = self.inception3a(x)
		x = self.inception3b(x)
		x = self.maxpool(x)

		x = self.inception4a(x)
		aux1_out = self.aux_head1(x) if self.training and self.use_aux and self.aux_head1 else None

		x = self.inception4b(x)
		x = self.inception4c(x)
		x = self.inception4d(x)
		aux2_out = self.aux_head2(x) if self.training and self.use_aux and self.aux_head2 else None

		x = self.inception4e(x)
		x = self.maxpool(x)

		x = self.inception5a(x)
		x = self.inception5b(x)

		x = self.global_pool(x)
		x = torch.flatten(x, 1)
		x = self.head_drop(x)
		logits = self.head(x)

		if self.training and self.use_aux:
			return logits, aux1_out, aux2_out
		return logits


if __name__ == "__main__":
	model = myGoogLeNet(num_classes=10, use_aux=True, drop_ratio=0.4)
	x = torch.randn(2, 3, 224, 224)

	model.train()
	train_out = model(x)
	print("myGoogLeNet train output shapes:", train_out[0].shape, train_out[1].shape, train_out[2].shape)

	model.eval()
	eval_out = model(x)
	print("myGoogLeNet eval output shape:", eval_out.shape)
