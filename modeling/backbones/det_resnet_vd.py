# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F
from .utils.conv_bn_layer import ConvBNLayer
from .utils.bottleneck_block import BottleneckBlock


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        shortcut=True,
        if_first=False,
    ):
        super().__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, act=None
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
            )

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv1
        y = F.relu(y)
        return y


class ResNet_vd(nn.Module):
    def __init__(
        self, in_channels=3, layers=50, dcn_stage=None, out_indices=None, **kwargs
    ):
        super().__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.dcn_stage = (
            dcn_stage if dcn_stage is not None else [False, False, False, False]
        )
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act="relu",
        )
        self.conv1_2 = ConvBNLayer(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, act="relu"
        )
        self.conv1_3 = ConvBNLayer(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, act="relu"
        )
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = []
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                is_dcn = self.dcn_stage[block]
                for i in range(depth[block]):
                    self.add_module(
                        "bb_%d_%d" % (block, i),
                        BottleneckBlock(
                            in_channels=(
                                num_channels[block]
                                if i == 0
                                else num_filters[block] * 4
                            ),
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            is_dcn=is_dcn,
                        ),
                    )
                    shortcut = True
                    block_list.append(self._modules["bb_%d_%d" % (block, i)])
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                block_list = []
                shortcut = False
                for i in range(depth[block]):
                    self.add_module(
                        "bb_%d_%d" % (block, i),
                        BasicBlock(
                            in_channels=(
                                num_channels[block] if i == 0 else num_filters[block]
                            ),
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                        ),
                    )
                    shortcut = True
                    block_list.append(self._modules["bb_%d_%d" % (block, i)])
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block])
                self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            if i in self.out_indices:
                out.append(y)
        return out

