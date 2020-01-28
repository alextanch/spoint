import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, inp_ch, out_ch):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Encoder(nn.Module):
    def __init__(self, inp_ch=1, ch1=64, ch2=64, ch3=128, out_ch=128):
        super().__init__()

        self.enc1 = EncoderBlock(inp_ch, ch1)
        self.enc2 = EncoderBlock(ch1, ch2)
        self.enc3 = EncoderBlock(ch2, ch3)
        self.enc4 = EncoderBlock(ch3, out_ch)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        return x


class Detector(nn.Module):
    def __init__(self, inp_ch=128, ch=256, out_ch=65):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Descriptor(nn.Module):
    def __init__(self, inp_ch=128, ch=256, out_ch=65):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_ch, ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.seq(x)

        norm = torch.norm(x, p=2, dim=1)
        desc = x.div(torch.unsqueeze(norm, 1))

        return desc


class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(inp_ch=1, ch1=64, ch2=64, ch3=128, out_ch=128)

        self.detector = Detector(inp_ch=128, ch=256, out_ch=65) if config.detector else None
        self.descriptor = Descriptor(inp_ch=128, ch=256, out_ch=256) if config.descriptor else None

    def forward(self, x):
        # encoder
        x = self.encoder(x)

        # detector head
        dets = self.detector(x) if self.detector else None

        # descriptor head
        desc = self.descriptor(x) if self.descriptor else None

        return dets, desc
