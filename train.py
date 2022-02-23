import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model.model import MLPModel

from option.config import Config
from trainer import train_epoch
# config file
config = Config({
    # device
    'gpu_id': "0",                          # specify GPU number to use
    'num_workers': 8,

    # data
    'db_name': 'AVA',                                     # database type
    'db_path': './dataset/train',      # root path of database
    'txt_file_name': './temp/AVA_scores.txt',                # list of images in the database
    'train_size': 0.8,                                          # train/vaildation separation ratio
    'scenes': 'all',                                            # using all scenes
    'batch_size': 8,
    'patch_size': 32,

    # optimization & training parameters
    'n_epoch': 100,                         # total training epochs
    'learning_rate': 1e-4,                  # initial learning rate
    'weight_decay': 0,                      # L2 regularization weight
    # 'momentum': 0.9,                        # SGD momentum
    'T_max': 3e4,                           # period (iteration) of cosine learning rate decay
    'eta_min': 0,                           # minimum learning rate
    'save_freq': 10,                        # save checkpoint frequency (epoch)
    'val_freq': 5,                          # validation frequency (epoch)


    # load & save checkpoint
    'weights_path': './weights',               # directory for saving checkpoint
    'checkpoint': None,                     # load checkpoint
})

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.FloatTensor(target)

# device setting
config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU %s' % config.gpu_id)
else:
    print('Using CPU')

# data selection
if config.db_name == 'AVA':
    from dataLoader.dataloader import AVADataset

# data load
train_dataset = AVADataset(txtFileName=config.txt_file_name)
val_dataset = AVADataset(test=True, txtFileName=config.txt_file_name)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=False,
    num_workers=config.num_workers, pin_memory=True, collate_fn =my_collate
    )
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False,
    num_workers=config.num_workers, pin_memory=True, collate_fn =my_collate
)

# create model
net = MLPModel()

# loss function & optimization
criterion = nn.L1Loss().to(config.device)
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, amsgrad=True, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# load weights & optimizer
if config.checkpoint is not None:
    checkpoint = torch.load(config.checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0

# make directory for saving weights
if not os.path.exists(config.weights_path):
    os.mkdir(config.weights_path)

# train & validation
for epoch in range(start_epoch, config.n_epoch):
    train_epoch(config, epoch, net, criterion, optimizer, scheduler, train_loader)