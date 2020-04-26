import torch
import torch.nn as nn
import torch.nn.init as init

class MattNet(nn.Module):
    def __init__(self):
        super(MattNet, self).__init__()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=13,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            dilation=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3),
            padding=2,
            dilation=2)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(3, 3),
            padding=4,
            dilation=4)
        self.conv4 = nn.Conv2d(
            in_channels=48,
            out_channels=16,
            kernel_size=(3, 3),
            padding=6,
            dilation=6)
        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=16,
            kernel_size=(3, 3),
            padding=8,
            dilation=8)
        self.conv6 = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=(3, 3),
            padding=1,
            dilation=1)
        self.interp = nn.UpsamplingBilinear2d(scale_factor=2)

        # feather
        self.convF1 = nn.Conv2d(
            in_channels=11,
            out_channels=8,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True)
        self.bn1 = nn.BatchNorm2d(8)
        self.convF2 = nn.Conv2d(
            in_channels=8,
            out_channels=3,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv5.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv6.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convF1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.convF2.weight, init.calculate_gain('relu'))

    def forward(self, x):
        """
        Segmentation Block network:

              input (x)
                |
          +-----------+
          |           |
        conv       max pool
         (a)         (b)
          |           |
          +-----------+
                |
          +---concat
          |    (c)
          |     |
          |   conv ------------+
          |    (d)             |
          |     |              |
          +-->concat           |
          +--- (e)             |
          |     |              |           interp (output)
          |   conv ------------+             |
          |    (f)             |           conv (l)
          |     |              |             |
          +-->concat           |---------> concat (k)
          +--- (g)             |
          |     |              |
          |   conv ------------+
          |    (h)             |
          |     |              |
          +-->concat           |
               (i)             |
                |              |
              conv ------------+
               (j)
        """
        a = self.relu(self.conv1(x))
        b = self.maxpool1(x)
        c = torch.cat((a, b), 1)

        d = self.relu(self.conv2(c))
        e = torch.cat((c, d), 1)

        f = self.relu(self.conv3(e))
        g = torch.cat((e, f), 1)

        h = self.relu(self.conv4(g))
        i = torch.cat((g, h), 1)

        j = self.relu(self.conv5(i))

        k = torch.cat((d, f ,h, j), 1)
        l = self.relu(self.conv6(k))
        output = self.interp(l)

        """
        Feathering Block:
        I, I.S, I.I -+
                     |- conv (fa) - BN (fb) - ReLU (fc) - conv (fd) - feathering
                     |                                                      |
        S -----------+------------------------------------------------------+
        """
        Sbg, Sfg = torch.split(output, 1, dim=1)

        I = x
        ISfg = I * Sfg
        II = I * I
        featherInput = torch.cat((I, ISfg, II, output), 1)
        fa = self.convF1(featherInput)
        fb = self.bn1(fa)
        fc = self.relu(fb)
        fd = self.convF2(fc)

        # feathering layer:
        A, B, C = torch.split(fd, 1, dim=1)
        alpha = A * Sfg + B * Sbg + C

        return alpha
