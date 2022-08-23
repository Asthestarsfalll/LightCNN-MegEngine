import math
import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import hub
from .pooling import MaxPool2d

class mfm(M.Module):
    CONV = 1
    LINEA = 2

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            mode=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if mode == mfm.CONV:
            self.filter = M.Conv2d(
                in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = M.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = F.split(x, 2, 1)
        return F.maximum(out[0], out[1])


class group(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels,
                        kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(M.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_9layers(M.Module):
    def __init__(self, num_classes=79077):
        super(network_9layers, self).__init__()
        self.features = M.Sequential(
            mfm(1, 48, 5, 1, 2),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(48, 96, 3, 1, 1),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = mfm(8*8*128, 256, mode=0)
        self.fc2 = M.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.dropout(x, drop_prob=0.2, training=self.training)
        out = self.fc2(x)
        return out, x


class network_29layers(M.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc = mfm(8*8*128, 256, mode=0)
        self.fc2 = M.Linear(256, num_classes)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return M.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.reshape(x.shape[0], -1)
        fc = self.fc(x)
        x = F.dropout(x, drop_prob=0.2, training=self.training)
        out = self.fc2(fc)
        return out, fc


class network_29layers_v2(M.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = M.Linear(8*8*128, 256)
        self.fc2 = M.Linear(256, num_classes, bias=False)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return M.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.reshape(x.shape[0], -1)
        fc = self.fc(x)
        x = F.dropout(x, drop_prob=0.2, training=self.training)
        out = self.fc2(x)
        return out, fc

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/95/files/b0fe2889-06e3-4a0a-b94f-1a90a4df11f1"
)
def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs) 
    return model

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/95/files/b57b6577-857e-4dda-ae70-49ba11057d59"
)
def LightCNN_29Layers(**kwargs):
    model = network_29layers(resblock, [1, 2, 3, 4], **kwargs)
    return model


def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model
