# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

import os
import sys
import numpy as np
import skimage
import signal
import random
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../..")))

import copy
from data.imaug import transform, create_operators
from data.simple_dataset import SimpleDataSet
# for PaddleX dataset_type
TextDetDataset = SimpleDataSet


__all__ = ["build_dataloader", "transform", "create_operators", "set_signal_handlers"]


def term_mp(sig_num, frame):
    """kill all child processes"""
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def set_signal_handlers():
    pid = os.getpid()
    try:
        pgid = os.getpgid(pid)
    except AttributeError:
        # In case `os.getpgid` is not available, no signal handler will be set,
        # because we cannot do safe cleanup.
        pass
    else:
        # XXX: `term_mp` kills all processes in the process group, which in
        # some cases includes the parent process of current process and may
        # cause unexpected results. To solve this problem, we set signal
        # handlers only when current process is the group leader. In the
        # future, it would be better to consider killing only descendants of
        # the current process.
        if pid == pgid:
            # support exit using ctrl+c
            signal.signal(signal.SIGINT, term_mp)
            signal.signal(signal.SIGTERM, term_mp)


def build_dataloader(config, mode, device, logger, seed=None, return_sampler=False):
    config = copy.deepcopy(config)
    module_name = config[mode]["dataset"]["name"]
    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]["loader"]
    batch_size = loader_config["batch_size_per_card"]
    drop_last = loader_config["drop_last"]
    shuffle = loader_config["shuffle"]
    num_workers = loader_config["num_workers"]

    if config['Global']['distributed']:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    data_loader = DataLoader(dataset, batch_size=batch_size,sampler=sampler, drop_last = drop_last,
                    shuffle=shuffle if sampler is None else False,num_workers=num_workers)
    if return_sampler:
        return data_loader, sampler
    return data_loader
