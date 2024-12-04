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
from __future__ import unicode_literals
import copy
import paddle

__all__ = ["build_optimizer"]


def build_lr_scheduler(optimizer,lr_config, epochs, step_each_epoch):
    from . import lr_scheduler

    lr_config.update({"epochs": epochs, "iters": step_each_epoch})
    lr_name = lr_config.pop("name", "CosineDecayLR")
    lr = getattr(lr_scheduler, lr_name)(optimizer=optimizer,**lr_config)
    return lr


def build_optimizer(config, epochs, step_each_epoch, model):
    from torch import optim
    optim_cfg = config['optimizer']
    optim_name = optim_cfg.pop("name", "Adam")
    optimizer = getattr(optim,optim_name)(model.parameters(), **optim_cfg)
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'], epochs, step_each_epoch)
    return optimizer, lr_scheduler

