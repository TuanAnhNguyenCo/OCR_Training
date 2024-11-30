import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer_config, loss_fn,
                 epochs, lr_scheduler=None, grad_accum_steps=1,device = 'cpu', train_sampler=None,
                 checkpoint_dir='./checkpoints', distributed=False, log_interval=10, logger = None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_config = optimizer_config
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = checkpoint_dir
        self.distributed = distributed
        self.log_interval = log_interval
        self.logger = logger
        self.device = device
        self.train_sampler = train_sampler
        os.makedirs(self.checkpoint_dir, exist_ok=True)


        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.optimizer_config['lr']['learning_rate'],
                                         betas=(self.optimizer_config['beta1'], self.optimizer_config['beta2']),weight_decay=float(self.optimizer_config['regularizer']['factor']))

        # LR Scheduler
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs - self.optimizer_config['lr']['warmup_epoch'])

        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            if self.distributed:
                batch = [b.to(self.local_rank) for b in batch]
            else:
                batch = [b.to(self.device) for b in batch]
            output = self.model(batch[0])
            loss = self.loss_fn(output, batch)
            optimized_loss = loss["loss"]
            # L2 Regularization
            # l2_loss = 0.0
            # for param in self.model.parameters():
            #     l2_loss += torch.norm(param, 2)
            # print(self.l2_reg)
            # optimized_loss += self.l2_reg * l2_loss

            optimized_loss = optimized_loss / self.grad_accum_steps
            optimized_loss.backward()
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_loss += optimized_loss.item() * batch[0].shape[0]

            if batch_idx % self.log_interval == 0:
                if self.distributed and dist.get_rank() == 0:
                    self.logger.info(f"Train Epoch: {epoch+1} [{batch_idx * batch[0].shape[0]}/{len(self.train_loader.dataset)} "
                          f"({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss}")
                elif not self.distributed:
                    self.logger.info(f"Train Epoch: {epoch+1} [{batch_idx * batch[0].shape[0]}/{len(self.train_loader.dataset)} "
                          f"({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss}")

        return train_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                if self.distributed:
                    batch = [b.to(self.local_rank) for b in batch]
                else:
                    batch = [b.to(self.device) for b in batch]
                output = self.model(batch[0])
                loss = self.loss_fn(output, batch)
                val_loss += loss.item() * batch[0].shape[0]
        return val_loss / len(self.val_loader.dataset)

    def save_checkpoint(self, epoch, val_loss):
        if self.distributed and dist.get_rank() != 0:
            return

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'checkpoint.pth'))

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']

    def train(self):
        start_epoch = 0
        if os.path.exists(os.path.join(self.checkpoint_dir, 'checkpoint.pth')):
            start_epoch = self.load_checkpoint(os.path.join(self.checkpoint_dir, 'checkpoint.pth'))

        # Warmup
        for epoch in range(start_epoch, self.optimizer_config['lr']['warmup_epoch']):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            if self.distributed:
                dist.reduce(torch.tensor(val_loss, device=self.local_rank), 0)
                if dist.get_rank() == 0:
                    val_loss = val_loss.item() / dist.get_world_size()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

            if self.distributed and dist.get_rank() == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            elif not self.distributed:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # CosineAnnealingLR
        for epoch in range(self.optimizer_config['lr']['warmup_epoch'], self.epochs):
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            if self.distributed:
                dist.reduce(torch.tensor(val_loss, device=self.local_rank), 0)
                if dist.get_rank() == 0:
                    val_loss = val_loss.item() / dist.get_world_size()

            self.lr_scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

            if self.distributed and dist.get_rank() == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            elif not self.distributed:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if self.distributed:
            dist.destroy_process_group()