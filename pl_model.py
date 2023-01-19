import inspect

import pytorch_lightning as pl
import torch
import numpy as np

from utils import VisActionSeg


class BasePLModel(pl.LightningModule):
    def __init__(self, torch_model: torch.nn.Module, optimizer_config: Dict,
                 lr_scheduler_config: Dict, loss_config: Dict, num_class: int):
        super().__init__()
        self.torch_model = torch_model  # Pytorch model
        self.optimizer_name = optimizer_config.pop('name')
        self.optimizer = create_optimizer(self.optimizer_name)

        self.lr = lr_scheduler_config.pop('lr')
        self.lr_scheduler_name = lr_scheduler_config.pop('name')
        self.lr_scheduler = create_lr_scheduler(self.lr_scheduler_name)
        self.lr_scheduler_config = lr_scheduler_config

        self.loss_name = loss_config.pop('name')
        self.loss_func = create_loss(self.loss_name)
        self.num_class = num_class

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_config)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        input, target = train_batch
        pred = self.model(input)
        loss = self.loss_func(pred, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, target = val_batch
        pred = self.model(input)
        loss = self.loss_func(pred, target)
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input, target = batch
        pred = self.model(input)
        return pred


class PlModel(pl.LightningModule):
    def __init__(self,
                 model, optimizer, lr, lr_scheduler, lr_scheduler_config,
                 loss_func, num_classes, action_dict, sample_rate, results_dir):
        super().__init__()

        # self.__dict__['_modules'] = model.__dict__['_modules']
        self.model = model  # Pytorch model
        self.optimizer = optimizer
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_config = lr_scheduler_config
        self.loss_func = loss_func
        self.num_classes = num_classes
        self.action_to_id = action_dict
        self.id_to_action = {
            idx: action for action, idx in self.action_to_id.items()}

        self.sample_rate = sample_rate
        self.results_dir = results_dir
        self.save_hyperparameters()

    # def on_save_checkpoint(self, checkpoint):
    #     checkpoint['PG'] = checkpoint['model'].PG
    #     checkpoint['Rs'] = checkpoint['model'].Rs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_config)
        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        input, target, mask, _ = train_batch
        pred = self.model(input)
        loss = self.loss_func(target, pred, mask)

        # epoch_loss += loss.item()

        # _, pred = torch.max(pred[-1].data, 1)
        # correct += ((pred == target).float()*mask[:, 0, :].squeeze(1)).sum().item()
        # total += torch.sum(mask[:, 0, :]).item()

        # batch_gen.reset()
        # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
        # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
        # logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
        # 													float(correct)/total))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, target, mask, _ = val_batch
        pred = self.model(input)
        loss = self.loss_func(target, pred, mask)
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input, _, _, vid = batch
        ms_pred = self.model(input)
        _, pred = torch.max(ms_pred[-1].data, 1)

        pred = pred.squeeze()
        recognition = []
        for action_idx in pred:
            # action = [list(self.id_to_action.keys())[list(
            #     self.id_to_action.values()).index(action_idx.item())]]*self.sample_rate
            recognition.extend(
                [self.id_to_action[action_idx.item()]] * self.sample_rate)

        f_name = vid[0].split('/')[-1].split('.')[0]
        with open(f'{self.results_dir}/{f_name}.txt', "w+") as fw:
            for pred_action in recognition:
                fw.write(f'{pred_action}\n')

        return pred
