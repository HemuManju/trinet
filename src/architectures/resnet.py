import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, output_size, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, output_size)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


class BasicBlockEncoder(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, downsample_func=None, stride=1):
        super(BasicBlockEncoder, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Thrid convolution
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample_func = downsample_func
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)

        # Downsample if needed
        if self.downsample_func is not None:
            identity = self.downsample_func(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class BasicBlockDecoder(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, scale_factor=4, stride=1):
        super().__init__()

        # First convolution
        self.upsample = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=scale_factor),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Thrid convolution
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        identity = x.clone()
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        x += identity
        return x


class ResNetEnc(nn.Module):
    def __init__(self, config, num_blocks=[1, 1, 1, 1], z_dim=10, nc=1):
        super().__init__()

        self.config = config
        self.latent_size = config['latent_size']

        self.in_channels = 16

        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlockEncoder, num_blocks[0], 16, stride=4)
        self.layer2 = self._make_layer(BasicBlockEncoder, num_blocks[1], 32, stride=4)
        self.layer3 = self._make_layer(BasicBlockEncoder, num_blocks[1], 64, stride=2)
        self.layer4 = self._make_layer(BasicBlockEncoder, num_blocks[1], 128, stride=2)
        self.linear = nn.Linear(128, z_dim)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        downsample_func = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            downsample_func = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, downsample_func=downsample_func, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        embeddings = self.linear(x)
        return embeddings


class ResNetDec(nn.Module):
    def __init__(self, config, num_blocks=[1, 1, 1, 1], z_dim=10, nc=1):
        super().__init__()
        self.config = config

        self.in_channels = 128
        n_classes = config['n_classes']

        self.latent_size = config['latent_size']
        self.image_size = config['image_resize']

        self.linear = nn.Linear(z_dim, 128)

        self.layer4 = self._make_layer(BasicBlockDecoder, num_blocks[3], 128, stride=2)
        self.layer3 = self._make_layer(BasicBlockDecoder, num_blocks[2], 64, stride=1)
        self.layer2 = self._make_layer(BasicBlockDecoder, num_blocks[1], 32, stride=1)
        self.layer1 = self._make_layer(
            BasicBlockDecoder, num_blocks[0], n_classes, stride=1
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def _make_layer(self, ResBlock, blocks, planes, stride):
        layers = []
        layers.append(ResBlock(self.in_channels, planes, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), self.latent_size, 1, 1)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        out = self.layer1(x)
        return self.relu(out)


def ResNet50(output_size, channels=1):
    return ResNet(Bottleneck, [1, 1, 1, 1], output_size, channels)


def ResNet101(output_size, channels=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], output_size, channels)


def ResNet152(output_size, channels=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], output_size, channels)


def SimpleResNet(output_size, channels=1):
    return ResNet(Bottleneck, [1, 1, 1, 1], output_size, channels)
