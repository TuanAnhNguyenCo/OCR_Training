from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import torch.distributed as dist
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
from utils.utility import set_seed
from program import preprocess
import logging
from det_trainer import DetectionDataModule, DetectionModule
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
import lightning as L


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


if __name__ == "__main__":
    logger = create_logger("train", "train.log", logging.INFO)
    config = preprocess(is_train=True)
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    data = DetectionDataModule(config=config, logger=logger)
    model = DetectionModule(config=config,train_steps=len(data.train_dataloader()))
    checkpoint_callback = ModelCheckpoint(dirpath="./", save_top_k=1, monitor="hmean")
    trainer = L.Trainer(default_root_dir="./", callbacks=[checkpoint_callback, StochasticWeightAveraging(swa_lrs=1e-2)],precision="bf16",
                accumulate_grad_batches=min(1, 64//config['Train']['loader']['batch_size_per_card']) if config['Global']["grad_accum_steps"] == True else 1,
                )
    trainer.fit(model,data)