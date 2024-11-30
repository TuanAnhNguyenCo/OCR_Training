# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import math
from torch.functional import F
import torch
from torch import nn


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        if_act=True,
        act=None,
    ):
        super().__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(
                    "The activation function({}) is selected incorrectly.".format(
                        self.act
                    )
                )
                exit()
        return x



class Head(nn.Module):
    def __init__(self, in_channels, kernel_list=[3, 2, 2], fix_nan=False, **kwargs):
        super(Head, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
        )
        self.conv_bn1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels // 4),
            nn.ReLU()
        )

        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
        )
        self.conv_bn2 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels // 4),
            nn.ReLU()
        )
        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2
        )

        self.fix_nan = fix_nan

        for layer in self.modules(): 
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x, return_f=False):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        if self.fix_nan and self.training:
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        if self.fix_nan and self.training:
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {"maps": shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.concat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return {"maps": y}


class LocalModule(nn.Module):
    def __init__(self, in_c, mid_c, use_distance=True):
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act="relu")
        self.last_1 = nn.Conv2d(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        outf = torch.concat([init_map, x], axis=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out


class PFHeadLocal(DBHead):
    def __init__(self, in_channels, k=50, mode="small", **kwargs):
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        self.mode = mode

        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest")
        if self.mode == "large":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif self.mode == "small":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)

    def forward(self, x, targets=None):
        shrink_maps, f = self.binarize(x, return_f=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None)
        cbn_maps = F.sigmoid(cbn_maps)
        if not self.training:
            return {"maps": 0.5 * (base_maps + cbn_maps), "cbn_maps": cbn_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.concat([cbn_maps, threshold_maps, binary_maps], dim=1)
        return {"maps": y, "distance_maps": cbn_maps, "cbn_maps": binary_maps}
