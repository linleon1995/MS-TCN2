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

from model import Trainer
from batch_gen import BatchGenerator, VAS_Dataset, read_data
from model import MS_TCN2


def align_video_length(batch):
    print(batch)
    seq_length_list = [sample['seq_length'] for sample in batch]
    N = len(batch) # batch_size
    C = batch[0]['input'].shape[0] # feature size
    L = max(seq_length_list) # sequence length
    K = batch[0]['mask'].shape[0] # number of class

    batch_input = torch.zeros([N, C, L])
    batch_target = -100 * torch.ones([N, L])
    batch_mask = torch.zeros([N, K, L])
    for idx, sample in enumerate(batch):
        sample_length = seq_length_list[idx]
        batch_input[idx, :, :sample_length] = torch.from_numpy(np.array(sample['input'], np.float32))
        batch_target[idx, :sample_length] = torch.from_numpy(np.array(sample['target'], np.int64))
        batch_mask[idx, :, :sample_length] = torch.from_numpy(np.array(sample['mask'], np.float32))
    
    return batch_input, batch_target, batch_mask


    
# XXX: valid dataset
# XXX: inference?
# XXX: loss
class PlModel(pl.LightningModule): 
	def __init__(self, model, optimizer, lr, loss_func):
		super().__init__()
		self.model = model
		self.optimizer = optimizer
		self.lr = lr
		self.loss_func = loss_func
		
	def configure_optimizers(self):
		return self.optimizer(self.parameters(), lr=self.lr)
		
	def training_step(self, train_batch, batch_idx):
		input, target, mask = train_batch
		input = input.view(input.size(0), -1)
		pred = self.model(input)
		loss = self.loss_func(target, pred)
		self.log('train_loss', loss)
		return loss
		
	def validation_step(self, val_batch, batch_idx):
		input, target, mask = val_batch
		input = input.view(input.size(0), -1)
		pred = self.model(input)
		loss = self.loss_func(target, pred)
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
    parser.add_argument('--dataset', default="breakfast")
    parser.add_argument('--split', default='1')

    parser.add_argument('--features_dim', default='2048', type=int)
    parser.add_argument('--bz', default='4', type=int)
    parser.add_argument('--lr', default='0.0005', type=float)


    parser.add_argument('--num_f_maps', default='64', type=int)

    # Need input
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_layers_PG', type=int, default=11)
    parser.add_argument('--num_layers_R', type=int, default=10)
    parser.add_argument('--num_R', type=int, default=3)

    parser.add_argument('--patience', type=int, default=20)

    args = parser.parse_args()

    num_epochs = args.num_epochs
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr

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


    # # data
    # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # train_loader = DataLoader(mnist_train, batch_size=32)
    # val_loader = DataLoader(mnist_val, batch_size=32)


    # XXX: valid dataset
    list_of_samples = read_data(vid_list_file)
    dataset = VAS_Dataset(
        num_classes, actions_dict, gt_path, features_path, sample_rate, list_of_samples)
    train_loader = DataLoader(dataset=dataset, collate_fn=align_video_length, batch_size=bz)

    valid_dataset = VAS_Dataset(
        num_classes, actions_dict, gt_path, features_path, sample_rate, list_of_samples)
    valid_loader = DataLoader(dataset=valid_dataset, collate_fn=align_video_length, batch_size=bz)


    # # data
    # train_data = VAS_Dataset(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)
    # # valid_data = VAS_Dataset(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)

    # batch_sampler = SequenceAlignSampler(batch_size=bz)
    # train_loader = DataLoader(train_data, batch_sampler=batch_sampler)
    # # val_loader = DataLoader(train_data, batch_sampler=batch_sampler)

	# model
    model = MS_TCN2(
        num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, 
        num_classes
    )
    # XXX: model generating factory
    optimizer = optim.Adam
    model = PlModel(model, optimizer, lr)

    
    # training
    # XXX: pl.Trainer arguments
    # TODO: checkpoint
    # TODO: pre-trained
    # TODO: restore
    trainer = pl.Trainer(
		gpus=[0], 
		precision=16, 
		limit_train_batches=0.5,
		callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),

        ]
	)
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=valid_loader
    )

if __name__ == '__main__':
    main()