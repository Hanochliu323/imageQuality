import os
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
import math


def train_epoch(config, epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    # 记录一个epoch的loss之和
    loss_sigma = 0.0
    total = 0.0
    plccTotal = 0.0
    net.train()

    for it, (inputs, targets)  in enumerate(train_loader):
        outputs = torch.from_numpy(np.array([]))
        targets = targets.reshape(targets.shape[0], 1)
        for i in range(len(inputs)):
            input = inputs[i]
            input = torch.FloatTensor(input)
            cutImgs = cutImg(input)
            a = np.array([])
            result = torch.from_numpy(a)
            for j in range(len(cutImgs)):
                cutImgBlock = cutImgs[j]
                cutImgBlock = cutImgBlock.to(config.device)
                cutImgBl = cutImgBlock.reshape(1, 3, 256, 256)
                output = net(cutImgBl)
                result = torch.cat((result, output), 0)
            meanResult = torch.mean(result, dim=0, keepdim=True)
            outputs = torch.cat((outputs, meanResult), 1)
        outputs = outputs.reshape(targets.shape[0], 1)
        targets = targets.to(config.device)

        # weight update
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total += targets.size(0)
        p = pearsonr(np.squeeze(outputs), np.squeeze(targets))
        loss_sigma += loss.item()
        plccTotal += p[0]

        print("[Training]: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Concern:{:.4f}".format(
            epoch + 1, config.n_epoch, it + 1, len(train_loader), loss_sigma/(it+1), plccTotal/(it+1)))

    # save weights
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    if (epoch + 1) % config.save_freq == 0:
        weights_file_name = time_str + "_epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))


def cutImg(img):
    imgs = []
    width = img.shape[1]
    depth = img.shape[2]
    wid = math.ceil(width / 256)
    dep = math.ceil(depth / 256)
    for i in range(wid):
        for j in range(dep):
            a = i * 256
            b = j * 256
            c = (i+1) * 256
            d = (j+1) * 256
            imgCut = img[:, a:c, b:d]
            if imgCut.shape[1] != 256:
                l = 256 - imgCut.shape[1]
                t = T.Pad(padding=[0, 0, 0, l], fill=0, padding_mode='constant')
                imgCut = t(imgCut)
            if imgCut.shape[2] != 256:
                l = 256 - imgCut.shape[2]
                t = T.Pad(padding=[0, 0, l, 0], fill=0, padding_mode='constant')
                imgCut = t(imgCut)

            imgs.append(imgCut)
    return imgs