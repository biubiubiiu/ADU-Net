import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class SimChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return self.model(x)


class Fusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.refine = nn.Conv2d(2 * channels, channels, 1, 1, 0, bias=False)
        self.attention = SimChannelAttention(2 * channels, channels)

    def forward(self, x, y):
        # assume y is smaller
        y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)
        out = F.relu(self.refine(torch.cat([x, y], dim=1)) + y, inplace=True)
        out = (1 + self.attention(torch.cat([x, y], dim=1))) * out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()

        padding = dilation
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.shortcut = weight_norm(nn.Conv2d(channels, channels, 1, 1, 0))

    def forward(self, x):
        return self.body(x) + self.shortcut(x)


class MultiDilatedMemBlock(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()

        dilation = lambda i: 1 + (i + 1) // 2
        self.resblocks = nn.ModuleList(
            [ResidualBlock(channels, dilation=dilation(i)) for i in range(num_blocks)]
        )
        self.reduction = weight_norm(
            nn.Conv2d(channels * num_blocks, channels, 1, 1, 0)
        )

    def forward(self, x):
        features = []
        for m in self.resblocks:
            x = m(x)
            features.append(x)

        x = torch.cat(features, dim=1)
        x = self.reduction(x)
        return x


class LLIE(nn.Module):
    def __init__(self):

        super(LLIE, self).__init__()

        self.conv1x = nn.Sequential(
            nn.Conv2d(4, 12, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4x = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv16x = nn.Sequential(
            nn.Conv2d(512, 128, 3, stride=1, padding=1, groups=128),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.res1x = nn.Conv2d(
            12, 12, kernel_size=3, stride=1, padding=1, padding_mode='reflect'
        )
        self.res4x = MultiDilatedMemBlock(32, num_blocks=3)
        self.res16x = MultiDilatedMemBlock(64, num_blocks=5)

        self.up16x4 = nn.Conv2d(64, 32, 3, 1, 1)
        self.up4x1 = nn.Conv2d(32, 12, 3, 1, 1)

        self.fusion16x4 = Fusion(32)
        self.fusion4x1 = Fusion(12)

    def forward(self, raw):
        feat1x = self.conv1x(raw)
        feat4x = self.conv4x(F.pixel_unshuffle(feat1x, 4))  # 32 channels
        feat16x = self.conv16x(F.pixel_unshuffle(feat4x, 4))  # 64 channels

        feat16x = self.res16x(feat16x)

        feat4x = self.fusion16x4(feat4x, self.up16x4(feat16x))
        feat4x = self.res4x(feat4x)

        feat1x = self.fusion4x1(feat1x, self.up4x1(feat4x))
        out = self.res1x(feat1x)

        out = F.pixel_shuffle(out, 2)
        out = torch.clamp(out, min=0.0, max=1.0)
        return out


if __name__ == '__main__':
    net = LLIE()
    inp = torch.randn(1, 4, 256, 256)
    print(net(inp).shape)
