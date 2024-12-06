import torch
import pytorch_warmup as warmup

class CosineDecayLR:
    def __init__(self, optimizer, epochs, eta_min=1e-6, warmup_epoch = 0, iters = 0):
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min)
        self.warmup_scheduler = warmup.LinearWarmup(optimizer, iters*warmup_epoch)
        self.iters = iters
        self.warmup_epoch = warmup_epoch
        self.optimizer = optimizer
    
    def step(self,epoch = None,idx = None):
        self.epoch = epoch
        assert epoch is not None and idx is not None
        if self.warmup_epoch == 0:
            self.lr_scheduler.step(epoch + idx / self.iters)
        else:
            with self.warmup_scheduler.dampening():
                if epoch >= self.warmup_epoch:
                    self.lr_scheduler.step(epoch - self.warmup_epoch + idx / self.iters)

    def state_dict(self):
        return {
            'epoch': self.epoch, 
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']