import os
import argparse
from pickletools import optimize
import random
from pathlib import Path

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

# from model import Trainer
from batch_gen import BatchGenerator, VAS_Dataset, read_data, get_vas_dataloader, get_train_dataloader
from model import MS_TCN2
from utils import mstcn_loss
from pl_model import PlModel


# TODO: Docstring for this repo.
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='predict')
    parser.add_argument('--dataset', default="coffee_room")
    parser.add_argument('--split', default='4')
    parser.add_argument(
        '--checkpoint', default='my/path/epoch=40-val_loss=1.60.ckpt')

    parser.add_argument('--features_dim', default='2048', type=int)
    parser.add_argument('--bz', default=4, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--lr', default='0.0005', type=float)

    parser.add_argument('--num_f_maps', default='64', type=int)

    # Need input
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_layers_PG', type=int, default=11)
    parser.add_argument('--num_layers_R', type=int, default=10)
    parser.add_argument('--num_R', type=int, default=4)

    parser.add_argument('--patience', type=int, default=40)

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

    vid_list_file = "./data/"+args.dataset + \
        "/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = "./data/"+args.dataset + \
        "/splits/test.split"+args.split+".bundle"
    features_path = r"C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door_feature\x3d_m/"
    # features_path = "./data/"+args.dataset+"/features/"
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

    # data
    # files = read_data(vid_list_file)
    # train_loader, valid_loader = get_train_dataloader(
    #     files=files,
    #     num_classes=num_classes,
    #     actions_dict=actions_dict,
    #     gt_path=gt_path,
    #     features_path=features_path,
    #     sample_rate=sample_rate,
    #     batch_size=bz,
    #     num_workers=num_workers,
    #     seed=seed
    # )

    train_files = read_data(vid_list_file)
    train_loader = get_vas_dataloader(
        train_files, num_classes, actions_dict, gt_path, features_path,
        sample_rate, bz, num_workers
    )
    test_files = read_data(vid_list_file_tst)
    valid_loader = get_vas_dataloader(
        test_files, num_classes, actions_dict, gt_path, features_path,
        sample_rate, 1, num_workers
    )

    # TODO: to find the not exist feature, reuse this code piece
    # from pathlib import Path
    # fs = list(Path(features_path).rglob('*.npy'))
    # fs_names = [f.stem for f in fs]
    # for t in train_files:
    #     t = Path(t)
    #     if t.stem not in fs_names:
    #         print(t.stem)

    # model
    vas_model = MS_TCN2(
        num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim,
        num_classes
    )

    # XXX: model generating factory
    optimizer = optim.Adam
    lr_scheduler = optim.lr_scheduler.StepLR
    lr_scheduler_config = {
        'gamma': 0.8,
        'step_size': 10
    }

    model = PlModel(
        model=vas_model,
        optimizer=optimizer,
        lr=lr,
        lr_scheduler=lr_scheduler,
        lr_scheduler_config=lr_scheduler_config,
        loss_func=mstcn_loss,
        num_classes=num_classes,
        action_dict=actions_dict,
        sample_rate=sample_rate,
        results_dir=results_dir,
    )
    # print(inspect.getmro(model))

    # training
    # XXX: pl.Trainer arguments
    # TODO: checkpoint
    # TODO: pre-trained
    # TODO: restore
    # TODO: predict

    mlf_logger = MLFlowLogger(
        experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
    mlflow.pytorch.autolog()

    if args.action == 'train':
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[0],
            precision=32,
            callbacks=[
                RichProgressBar(theme=RichProgressBarTheme(
                    progress_bar="green")),
                EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience),
                ModelCheckpoint(
                    dirpath="my/path/",
                    filename='{epoch}-{val_loss:.2f}',
                    save_top_k=1,
                    monitor="val_loss",
                    save_weights_only=True,
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
    elif args.action == 'predict':
        # pred
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=[0],
            precision=32,
            callbacks=[
                RichProgressBar(theme=RichProgressBarTheme(
                    progress_bar="green")),
            ],
            logger=mlf_logger,
        )

        files = read_data(vid_list_file_tst)
        test_loader = get_vas_dataloader(
            files, num_classes, actions_dict, gt_path, features_path,
            sample_rate, 1, num_workers
        )
        model = PlModel.load_from_checkpoint(args.checkpoint)
        model.eval()
        trainer.predict(
            model=model,
            dataloaders=test_loader
        )


if __name__ == '__main__':
    # main()

    # model_uri = './mlruns/0/778692b74cc2437b97f70fa0f80734f0/artifacts/model'
    model_uri = r'C:\Users\test\Desktop\Leon\Projects\MS_TCN2\results_old\9b7114c27f9440c1b1d6f7836e8d18c0/artifacts/model'

    pytorch_pyfunc = mlflow.pyfunc.load_model(model_uri=model_uri)

    f = Path(r'C:\Users\test\Desktop\Leon\Datasets\coffee_room\events_door_feature\x3d_m\20221129080000_coffee_video_111995_112445.npy')
    video_data = np.load(f)
    video_data = np.expand_dims(video_data, 0)

    pred = pytorch_pyfunc.predict(video_data)
    pass
