import torch
from torch import nn

def conv3X3(in_channels, out_channels, stride=1, pad=1):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=pad, bias=False)

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3X3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3X3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsampel = downsample
        # self.fc1 = nn.Linear(,2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampel:
            residual = self.downsampel(x)
        out += residual
        return self.relu(out)

class UnetSegment(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(UnetSegment, self).__init__()

        self.in_channels = in_channels
        self.block = ResBlock
        self.d_layer1 = self.make_layer(1, 32)
        self.d_layer2 = self.make_layer(2, 64)
        self.d_layer3 = self.make_layer(3, 128)
        self.d_layer4 = self.make_layer(4, 256)
        self.d_layer5 = self.make_layer(5, 512)

        self.pooling = self.max_pool()

        self.deconv4 = self.de_conv(512, 256)
        self.u_layer4 = self.make_layer(4+2, 256)
        self.deconv3 = self.de_conv(256, 128)
        self.u_layer3 = self.make_layer(3+2, 128)
        self.deconv2 = self.de_conv(128, 64)
        self.u_layer2 = self.make_layer(2+2, 64)
        self.deconv1 = self.de_conv(64, 32)
        self.u_layer1 = self.make_layer(1+2, 32)

        self.conv = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)
        # self.fc = nn.Sequential(nn.Linear(25*25*512, 1024), nn.Linear(1024, 2)) # 400x400
        # self.fc = nn.Sequential(nn.Linear(16*16*512, 1024), nn.Linear(1024, 2)) # 256x256
        self.fc = nn.Sequential(nn.Linear(13*13*512, 1024), nn.Linear(1024, 2)) # 208x208
        # self.fc = nn.Sequential(nn.Linear(32*32*512, 1024), nn.Linear(1024, 2)) # 512x512

    def make_layer(self, layer, out_channels, stride=1):
        if layer > 1:
            self.in_channels = 2**(layer+3)
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3X3(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        return self.block(self.in_channels, out_channels, stride, downsample)


    def de_conv(self, in_channels, out_channels, stride=2, pad=1):
        return  nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=pad, output_padding=1, bias=False)

    def max_pool(self, kernel_size=2):
        return nn.MaxPool2d(kernel_size)

    def forward(self, x):
        out1 = self.d_layer1(x)
        out1_p = self.pooling(out1)
        out2 = self.d_layer2(out1_p)
        out2_p = self.pooling(out2)
        out3 = self.d_layer3(out2_p)
        out3_p = self.pooling(out3)
        out4 = self.d_layer4(out3_p)
        # print out4.size(1)
        out4_p = self.pooling(out4)
        out5 = self.d_layer5(out4_p)
        up4 = self.deconv4(out5)
        upout4 = self.u_layer4(torch.cat((out4, up4), 1))
        up3 = self.deconv3(upout4)
        upout3 = self.u_layer3(torch.cat((out3, up3), 1))
        up2 = self.deconv2(upout3)
        upout2 = self.u_layer2(torch.cat((out2, up2), 1))
        up1 = self.deconv1(upout2)
        upout1 = self.u_layer1(torch.cat((out1, up1), 1))

        return self.conv(upout1), self.fc(out5.view(out5.size(0), -1))

class UnetDetection(nn.Module):
    def __init__(self, block, in_channels, num_classes=2):
        super(UnetDetection, self).__init__()

        self.in_channels = in_channels
        self.block = block
        self.d_layer1 = self.make_layer(1, 32)
        self.d_layer2 = self.make_layer(2, 64)
        self.d_layer3 = self.make_layer(3, 128)
        self.d_layer4 = self.make_layer(4, 256)
        self.d_layer5 = self.make_layer(5, 512)

        self.pooling = self.max_pool()

        self.deconv4 = self.de_conv(512, 256)
        self.u_layer4 = self.make_layer(4+2, 256)
        self.deconv3 = self.de_conv(256, 128)
        self.u_layer3 = self.make_layer(3+2, 128)
        self.deconv2 = self.de_conv(128, 64)
        self.u_layer2 = self.make_layer(2+2, 64)
        self.deconv1 = self.de_conv(64, 32)
        self.u_layer1 = self.make_layer(1+2, 32)

        self.conv = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)
        self.detec_conv = conv3X3(32, 25)

    def make_layer(self, layer, out_channels, stride=1):
        if layer > 1:
            self.in_channels = 2**(layer+3)
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(conv3X3(self.in_channels, out_channels, stride=stride),
                                       nn.BatchNorm2d(out_channels))
        return self.block(self.in_channels, out_channels, stride, downsample)


    def de_conv(self, in_channels, out_channels, stride=2, pad=1):
        return  nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=pad, output_padding=1, bias=False)

    def max_pool(self, kernel_size=2):
        return nn.MaxPool2d(kernel_size)

    def forward(self, x):
        out1 = self.d_layer1(x)
        out1_p = self.pooling(out1)
        out2 = self.d_layer2(out1_p)
        out2_p = self.pooling(out2)
        out3 = self.d_layer3(out2_p)
        out3_p = self.pooling(out3)
        out4 = self.d_layer4(out3_p)
        # print out4.size(1)
        out4_p = self.pooling(out4)
        out5 = self.d_layer5(out4_p)
        up4 = self.deconv4(out5)
        upout4 = self.u_layer4(torch.cat((out4, up4), 1))
        up3 = self.deconv3(upout4)
        upout3 = self.u_layer3(torch.cat((out3, up3), 1))
        up2 = self.deconv2(upout3)
        upout2 = self.u_layer2(torch.cat((out2, up2), 1))
        up1 = self.deconv1(upout2)
        upout1 = self.u_layer1(torch.cat((out1, up1), 1)) # shape=[batch_size, 32, 256, 256]

        # return self.conv(upout1)  # segmentation

        # for dettection
        out = self.detec_conv(upout1)   # shape = [batch_size, 5x5, 256, 256]
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        out = out.transpose(1,2).contiguous().view(
            size[0], size[2], size[3], 5, 5)

        return out

