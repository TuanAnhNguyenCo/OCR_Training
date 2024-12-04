from .conv_bn_layer import ConvBNLayer
from torch import nn
from torch.nn import functional as F

class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        shortcut=True,
        if_first=False,
        is_dcn=False,
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
            is_dcn=is_dcn,
            dcn_groups=2,
        )
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
            )

        self.shortcut = shortcut

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