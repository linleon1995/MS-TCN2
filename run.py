import os
import argparse
from pickletools import optimize
import random
from h11 import Data

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from torch import optim
import numpy as np
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import mlflow.pytorch

from model import Trainer
from batch_gen import BatchGenerator, VAS_Dataset, read_data
from model import MS_TCN2


def mstcn_loss(batch_target, predictions, mask):
	cls_loss = 0
	smooth_loss = 0
	_lambda = 1 # classification loss weight
	_tau = 0.15 # smooth loss weight
	num_classes = predictions.shape[2]
	# TODO: why clamp to 0 16?
	for p in predictions:
		cls_loss += nn.CrossEntropyLoss(ignore_index=-100)(
			p.transpose(2, 1).contiguous().view(-1, num_classes), batch_target.view(-1))

		temp_smooth_loss = nn.MSELoss(reduction='none')(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1))
		smooth_loss += torch.mean(
			torch.clamp(
				temp_smooth_loss, min=0, max=16
			)*mask[:, :, 1:]
		)
	loss = _lambda*cls_loss + _tau*smooth_loss
	return loss


def align_video_length(batch):
    seq_length_list = [sample['seq_length'] for sample in batch]
    N = len(batch) # batch_size
    C = batch[0]['input'].shape[0] # feature size
    L = max(seq_length_list) # sequence length
    K = batch[0]['mask'].shape[0] # number of class

    batch_input = torch.zeros([N, C, L], dtype=torch.float)
    batch_target = -100 * torch.ones([N, L], dtype=torch.long)
    batch_mask = torch.zeros([N, K, L], dtype=torch.float)
    for idx, sample in enumerate(batch):
        sample_length = seq_length_list[idx]
        batch_input[idx, :, :sample_length] = torch.from_numpy(np.array(sample['input'], np.float32))
        batch_target[idx, :sample_length] = torch.from_numpy(np.array(sample['target'], np.int64))
        batch_mask[idx, :, :sample_length] = torch.from_numpy(np.array(sample['mask'], np.float32))
    
    return batch_input, batch_target, batch_mask


class PlModel(pl.LightningModule): 
	def __init__(self, model, optimizer, lr, lr_scheduler, lr_scheduler_config, loss_func, num_classes):
		super().__init__()
		self.model = model
		self.optimizer = optimizer
		self.lr = lr
		self.lr_scheduler = lr_scheduler
		self.lr_scheduler_config = lr_scheduler_config
		self.loss_func = loss_func
		self.num_classes = num_classes
		
	def configure_optimizers(self):
		optimizer = self.optimizer(self.parameters(), lr=self.lr)
		lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_config)
		return [optimizer], [lr_scheduler]
		
	def training_step(self, train_batch, batch_idx):
		input, target, mask = train_batch
		# input = input.view(input.size(0), -1)
		pred = self.model(input)
		loss = self.loss_func(target, pred, mask)
		
		# epoch_loss += loss.item()

		# _, predicted = torch.max(pred[-1].data, 1)
		# correct += ((predicted == target).float()*mask[:, 0, :].squeeze(1)).sum().item()
		# total += torch.sum(mask[:, 0, :]).item()



		# batch_gen.reset()
		# torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
		# torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
		# logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
		# 													float(correct)/total))
		self.log('train_loss', loss)
		return loss
		
	def validation_step(self, val_batch, batch_idx):
		input, target, mask = val_batch
		# input = input.view(input.size(0), -1)
		pred = self.model(input)
		loss = self.loss_func(target, pred, mask)
		self.log('val_loss', loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train')
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')

    parser.add_argument('--features_dim', default='2048', type=int)
    parser.add_argument('--bz', default=4, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default='0.0005', type=float)


    parser.add_argument('--num_f_maps', default='64', type=int)

    # Need input
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_layers_PG', type=int, default=11)
    parser.add_argument('--num_layers_R', type=int, default=10)
    parser.add_argument('--num_R', type=int, default=3)

    parser.add_argument('--patience', type=int, default=20)

    args = parser.parse_args()

    num_epochs = args.num_epochs
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr
    num_workers = args.num_workers

    num_layers_PG = args.num_layers_PG
    num_layers_R = args.num_layers_R
    num_R = args.num_R
    num_f_maps = args.num_f_maps

    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2

    vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = "./data/"+args.dataset+"/features/"
    gt_path = "./data/"+args.dataset+"/groundTruth/"

    mapping_file = "./data/"+args.dataset+"/mapping.txt"

    model_dir = "./models/"+args.dataset+"/split_"+args.split
    results_dir = "./results/"+args.dataset+"/split_"+args.split

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)

    patience = args.patience
    # trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split)
    # if args.action == "train":
    #     batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    #     batch_gen.read_data(vid_list_file)
    #     trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

    # if args.action == "predict":
    #     trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)


    # data
    train_samples = read_data(vid_list_file)
    dataset = VAS_Dataset(
        num_classes, actions_dict, gt_path, features_path, sample_rate, train_samples)
    train_loader = DataLoader(dataset=dataset, collate_fn=align_video_length, batch_size=bz, num_workers=num_workers)

    valid_samples = read_data(vid_list_file_tst)
    valid_dataset = VAS_Dataset(
        num_classes, actions_dict, gt_path, features_path, sample_rate, valid_samples)
    valid_loader = DataLoader(dataset=valid_dataset, collate_fn=align_video_length, batch_size=bz, num_workers=num_workers)


	# model
    model = MS_TCN2(
        num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, 
        num_classes
    )
    # XXX: model generating factory
    optimizer = optim.Adam
    lr_scheduler = optim.lr_scheduler.StepLR
    lr_scheduler_config = {
		'gamma': 0.8,
    	'step_size': 50
	}

    model = PlModel(
		model=model,
		optimizer=optimizer,
		lr=lr,
		lr_scheduler=lr_scheduler,
		lr_scheduler_config=lr_scheduler_config,
		loss_func=mstcn_loss,
		num_classes=num_classes
	)
    
    # training
    # XXX: pl.Trainer arguments
    # TODO: checkpoint
    # TODO: pre-trained
    # TODO: restore
	# TODO: predict
	
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
    mlflow.pytorch.autolog()
    trainer = pl.Trainer(
		gpus=[0], 
		precision=32,
		callbacks=[
            RichProgressBar(theme=RichProgressBarTheme(progress_bar="green")),
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
			ModelCheckpoint(
				dirpath="my/path/", 
				filename='{epoch}-{val_loss:.2f}', 
				save_top_k=1, 
				monitor="val_loss"
			),
        ],
		max_epochs=num_epochs,
		logger=mlf_logger,
	)
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader
    )

if __name__ == '__main__':
    main()