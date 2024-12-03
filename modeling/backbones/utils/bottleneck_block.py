from .conv_bn_layer import ConvBNLayer
from torch import nn
from torch.nn import functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, num_channels, num_filters, stride, shortcut=True, is_dcn=False):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            act="relu",
            is_dcn=is_dcn,
            dcn_groups=1,
        )
        self.conv2 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            act=None,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                stride=stride,
            )

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = short + conv2
        y = F.relu(y)
        return y