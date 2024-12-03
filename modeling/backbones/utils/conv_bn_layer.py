from torch import nn
from .deformable_conv import DeformableConv2d

class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        dcn_groups=1,
        is_vd_mode=False,
        act=None,
        is_dcn=False,
    ):
        super().__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )
        if not is_dcn:
            self._conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
            )
        else:
            self._conv = DeformableConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=dcn_groups,  # groups
            )
        self._batch_norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == 'relu' else nn.Identity()
        )

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y