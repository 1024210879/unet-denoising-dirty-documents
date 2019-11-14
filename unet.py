import torch.nn as nn
import torch


class Inception(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Inception, self).__init__()
        hide_ch = out_ch // 2
        self.inception = nn.Sequential(
            nn.Conv2d(in_ch, hide_ch, 1),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, hide_ch, 3, padding=1, groups=hide_ch),
            nn.BatchNorm2d(hide_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_ch, out_ch, 1)
        )

    def forward(self, x):
        return self.inception(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.doubleConv = nn.Sequential(
            Inception(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            Inception(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        # down
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.Conv2d(64, 64, 2, 2, groups=64)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.Conv2d(128, 128, 2, 2, groups=128)
        self.bottom = DoubleConv(128, 256)
        # up
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = DoubleConv(128 * 2, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = DoubleConv(64 * 2, 64)
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # down
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        bottom = self.bottom(pool2)
        # up
        up3 = self.up3(bottom)
        merge3 = torch.cat([up3, conv2], dim=1)
        conv3 = self.conv3(merge3)
        up4 = self.up4(conv3)
        merge4 = torch.cat([up4, conv1], dim=1)
        conv4 = self.conv4(merge4)
        out = self.out(conv4)
        return nn.Sigmoid()(out)


if __name__ == '__main__':
    net = UNet(1, 2)
    inputs = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    output = net(inputs)
    print(output.size())
    print(output)
