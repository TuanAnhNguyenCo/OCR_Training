import lightning as L
from modeling.architectures import build_model
from optimizer import build_optimizer
from losses import build_loss
from postprocess import build_post_process
from metrics import build_metric
from data import build_dataloader
import numpy as np
import torch


class DetectionModule(L.LightningModule):
    def __init__(self,config,train_steps):
        super().__init__()
        self.model = build_model(config["Architecture"])
        self.loss_class = build_loss(config["Loss"])
        self.post_process_func = build_post_process(config["PostProcess"])
        self.metric_class = build_metric(config["Metric"])
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.config = config
        self.train_steps = train_steps
        self.metric = {'precision': 0, 'recall': 0.0, 'hmean': 0}
        self.count = 0
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        output = self.model(batch[0])
        loss = self.loss_class(output, batch)
        avg_loss = loss["loss"] / self.trainer.accumulate_grad_batches
        self.manual_backward(avg_loss)
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
        self.log_dict(loss,prog_bar=True)
        sch = self.lr_schedulers()
        sch.step(self.trainer.current_epoch, batch_idx)
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        output = self.model(batch[0])
        output = {key:value.cpu().numpy().astype(np.float32) for key, value in output.items()}
        batch_numpy = []
        batch = [b.cpu() for b in batch]
        for item in batch:
            batch_numpy.append(item.numpy())
        post_result = self.post_process_func(output, batch_numpy[1])
        self.metric_class(post_result, batch_numpy)
        result = self.metric_class.get_metric()
        self.log("hmean",result["hmean"])
        self.metric = {key: value + self.metric[key] for key, value in result.items()}
        self.count += 1
        
    
    def on_validation_epoch_end(self):
        metric = {key: value / self.count for key, value in self.metric.items()}
        print("Validation Metric: ", metric)
        self.metric = {'precision': 0, 'recall': 0.0, 'hmean': 0}
        self.count = 0
        super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        output = self.model(batch[0])
        output = {key:value.cpu().numpy().astype(np.float32) for key, value in output.items()}
        batch_numpy = []
        batch = [b.cpu() for b in batch]
        for item in batch:
            batch_numpy.append(item.numpy())
        post_result = self.post_process_func(output, batch_numpy[1])
        self.metric_class(post_result, batch_numpy)
        self.log("hmean",self.metric_class.get_metric()["hmean"])
        self.metric = {key: value + self.metric[key] for key, value in self.metric_class.get_metric().items()}
        self.count += 1
    
    def on_test_epoch_end(self):
        metric = {key: value / self.count for key, value in self.metric.items()}
        print("Test Metric: ", metric)
        self.metric = {'precision': 0, 'recall': 0.0, 'hmean': 0}
        self.count = 0
        super().on_test_epoch_end()

    def configure_optimizers(self):
        optimizer, lr_scheduler = build_optimizer(
            config=self.config,
            epochs=self.config["Global"]["epoch_num"],
            step_each_epoch=self.train_steps,
            model=self.model,
        )
        return (
            {"optimizer": optimizer, "lr_scheduler": lr_scheduler},
        )

class DetectionDataModule(L.LightningDataModule):
    def __init__(self, config, logger):
        super().__init__()
        self.config = config
        self.logger = logger

    def train_dataloader(self):
        train_dataloader = build_dataloader(self.config, "Train", logger=self.logger, return_sampler=False)
        return train_dataloader

    def val_dataloader(self):
        self.config['Eval']['dataset']['label_file_list'] = [self.config['Eval']['dataset']['label_file_list'][0].replace("test.txt",'val.txt')]
        val_dataloader = build_dataloader(self.config, "Eval", logger=self.logger, return_sampler=False)
        return val_dataloader

    def test_dataloader(self):
        self.config['Eval']['dataset']['label_file_list'] = [self.config['Eval']['dataset']['label_file_list'][0].replace("val.txt",'test.txt')]
        test_dataloader = build_dataloader(self.config, "Eval", logger=self.logger, return_sampler=False)
        return test_dataloader
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = [i.to(device) for i in batch]
        return batch


if __name__ == '__main__':
    config = {}
    model = DetectionModule(config)
    trainer = L.Trainer(default_root_dir="./")
    trainer.fit(model)