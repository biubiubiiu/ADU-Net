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
        out = F.relu(self.refine(torch.cat([x, y], dim=1)) + x, inplace=True)
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


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor):
        super().__init__()

        in_channels = in_channels * down_factor**2

        if in_channels < 512:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            mid_channels = out_channels * 2
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, 1, 1, groups=mid_channels),
                nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            )

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.down = nn.PixelUnshuffle(down_factor)

    def forward(self, x):
        return self.act(self.conv(self.down(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, up_factor):
        super().__init__()

        self.trasform = MultiDilatedMemBlock(in_channels, n_blocks)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=up_factor),
        )
        self.fusion = Fusion(out_channels)

    def forward(self, x, y):
        x = self.trasform(x)
        out = self.fusion(self.up(x), y)
        return out


class LLIE(nn.Module):

    in_channels = 4
    out_channels = 3
    mid_channels = (12, 32, 64)
    n_blocks = (5, 3)
    scale_factor = 4

    def __init__(self):

        super(LLIE, self).__init__()

        # Initial projection
        self.conv0 = nn.Conv2d(self.in_channels, self.mid_channels[0], 3, 1, 1)

        # Downsample
        self.downs = nn.ModuleList(
            [
                DownBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i + 1],
                    down_factor=self.scale_factor,
                )
                for i in range(len(self.mid_channels) - 1)
            ]
        )

        # Upsample
        self.ups = nn.ModuleList(
            [
                UpBlock(
                    in_channels=self.mid_channels[-i - 1],
                    out_channels=self.mid_channels[-i - 2],
                    n_blocks=self.n_blocks[i],
                    up_factor=self.scale_factor,
                )
                for i in range(len(self.mid_channels) - 1)
            ]
        )

        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.mid_channels[0], self.out_channels * 4, kernel_size=3),
            nn.PixelShuffle(upscale_factor=2),
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)

        feats = [x]
        for down in self.downs:
            x = down(x)
            feats.append(x)

        x = feats.pop()
        for up in self.ups:
            enc_feat = feats.pop()
            x = up(x, enc_feat)

        out = self.out(x)
        out = torch.clamp(out, min=0.0, max=1.0)
        return out


if __name__ == '__main__':
    net = LLIE()
    inp = torch.randn(1, 4, 256, 256)
    print(net(inp).shape)
