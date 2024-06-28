import torch
import torch.nn as nn
import torch.nn.functional as F


class PDN_S(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=1, padding=3 * padding)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=3 * padding)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1 * padding)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4, stride=1, padding=0 * padding)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class PDN_M(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * padding)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * padding)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * padding)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * padding)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * padding)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * padding)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels: int, img_size: int, padding: bool = False) -> None:
        super().__init__()
        self.img_size = img_size
        self.last_upsample = img_size // 4 if padding else img_size // 4 - 8
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.img_size // 64 - 1, mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=self.img_size // 32, mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=self.img_size // 16 - 1, mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=self.img_size // 8, mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=self.img_size // 4 - 1, mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=self.img_size // 2 - 1, mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, out_channels: int, img_size: int, padding: bool = False) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, img_size, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
