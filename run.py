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

from model import Trainer
from batch_gen import BatchGenerator, VAS_Dataset, SequenceAlignSampler, WeightedFixedBatchSampler, read_data
from model import MS_TCN2

    
class PlModel(pl.LightningModule): 
	def __init__(self, model, optimizer, lr):
		super().__init__()
		self.model = model
		self.optimizer = optimizer
		self.lr = lr

	def forward(self, x):
		y = self.model(x)
		return y
		
	def configure_optimizers(self):
		return self.optimizer(self.parameters(), lr=self.lr)
		
	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss
		
	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='predict')
    parser.add_argument('--dataset', default="breakfast")
    parser.add_argument('--split', default='1')

    parser.add_argument('--features_dim', default='2048', type=int)
    parser.add_argument('--bz', default='1', type=int)
    parser.add_argument('--lr', default='0.0005', type=float)


    parser.add_argument('--num_f_maps', default='64', type=int)

    # Need input
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_layers_PG', type=int, default=11)
    parser.add_argument('--num_layers_R', type=int, default=10)
    parser.add_argument('--num_R', type=int, default=3)

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

    # XXX: Check sampler working
    # XXX: valid dataset
    list_of_samples = read_data(vid_list_file)
    dataset = VAS_Dataset(num_classes, actions_dict, gt_path, features_path, sample_rate, list_of_samples)
    sampler = RandomSampler(data_source=dataset)
    batch_sampler = WeightedFixedBatchSampler(
        sampler, batch_size=2, drop_last=True, list_of_samples=list_of_samples
    )
    train_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)


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
    trainer = pl.Trainer(gpus=[0], precision=16, limit_train_batches=0.5)
    trainer.fit(model, train_loader, train_loader)

if __name__ == '__main__':
    main()