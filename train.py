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
import torch

import os
import sys
import torch.distributed as dist
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import yaml
from data import build_dataloader, set_signal_handlers
from modeling.architectures import build_model
from losses import build_loss
from postprocess import build_post_process
from metrics import build_metric
from utils.save_load import load_model
from optimizer import build_optimizer
from utils.utility import set_seed
from program import preprocess
import logging
from trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP

def create_logger(name, log_file=None, log_level=logging.INFO):

  logger = logging.getLogger(name)
  logger.setLevel(log_level)

  # Tạo formatter
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  # Tạo console handler
  ch = logging.StreamHandler()
  ch.setLevel(log_level)
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  # Tạo file handler (optional)
  if log_file:
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

  return logger

def main(config,logger, device, seed):
    # init dist environment

    global_config = config["Global"]

    # build model
    model = build_model(config["Architecture"])
     # load pretrain model
    model = load_model(
        config, model
    )
    if global_config['torch_compile']:
        model = torch.compile(model, dynamic=True, mode="reduce-overhead")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    if global_config['distributed']:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        model = DDP(model.to(local_rank),
                              device_ids=[local_rank],
                              output_device=local_rank)
    # build dataloader
    set_signal_handlers()
    train_dataloader, train_sampler = build_dataloader(config, "Train", device, logger, seed, return_sampler=True)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n"
            + "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            + "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config["Eval"]:
        valid_dataloader = build_dataloader(config, "Eval", device, logger, seed)
    else:
        valid_dataloader = None

    # build loss
    loss_class = build_loss(config["Loss"])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    logger.info("train dataloader has {} iters".format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))
    post_process_func = build_post_process(config["PostProcess"])
    metric_class = build_metric(config["Metric"])

    trainer = Trainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=valid_dataloader,
        loss_fn=loss_class,
        epochs=global_config["epoch_num"],
        grad_accum_steps=min(1, 64//config['Train']['loader']['batch_size_per_card']) if global_config["grad_accum_steps"] == True else 1,
        distributed=global_config["distributed"],
        log_interval=global_config["print_batch_step"],
        logger=logger,
        device=device,
        train_sampler=train_sampler,
        post_process_func=post_process_func,
        metric_class = metric_class,
        eval_batch_step = global_config["eval_batch_step"],
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()

if __name__ == "__main__":
    logger = create_logger("train", "train.log", logging.INFO)
    config = preprocess(is_train=True)
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    main(config, logger, "mps", seed)