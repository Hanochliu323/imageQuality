import os
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
import math
from torchvision import transforms as T

def train_epoch(config, train_loader, epoch, net, criterion, optimizer, scheduler):
    losses = []

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    # 记录一个epoch的loss之和
    loss_sigma = 0.0
    # total = 0.0
    # plccTotal = 0.0
    net.train()

    for it, (inputs, labels)  in enumerate(train_loader):
        outputs = torch.from_numpy(np.array([])).to(config.device)
        labels = labels.reshape(labels.shape[0], 1).to(config.device)
        for i in range(len(inputs)):
            input = inputs[i]
            input = torch.FloatTensor(input).to(config.device)
            cutImgs = cutImg(input)
            a = np.array([])
            result = torch.from_numpy(a).to(config.device)
            for j in range(len(cutImgs)):
                cutImgBlock = cutImgs[j].to(config.device)
                # cutImgBlock = cutImgBlock.to(config.device)
                cutImgBl = cutImgBlock.reshape(1, 3, 256, 256).to(config.device)
                output = net(cutImgBl).to(config.device)
                result = torch.cat((result, output), 0).to(config.device)
            meanResult = torch.mean(result, dim=0, keepdim=True).to(config.device)
            outputs = torch.cat((outputs, meanResult), 1).to(config.device)
        outputs = outputs.reshape(labels.shape[0], 1).to(config.device)
        labels = labels.to(config.device)

        # weight update
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        pred_batch_numpy = outputs.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)


        rho_s2, _ = spearmanr(np.squeeze(pred_batch_numpy), np.squeeze(labels_batch_numpy))
        rho_p2, _ = pearsonr(np.squeeze(pred_batch_numpy), np.squeeze(labels_batch_numpy))
        print('--[trainIteration] epoch:%d | Iteration:%d/%d | loss:%f | SROCC:%4f | PLCC:%4f' % (epoch + 1,it + 1, len(train_loader), loss.item(), rho_s2, rho_p2))
        # # total += labels.size(0)
        # o = np.squeeze(outputs).data.cpu().numpy()
        # t = np.squeeze(labels).data.cpu().numpy()
        # rho_p, _ = pearsonr(o, t)
        # rho_s, _ = spearmanr(o, t)
        # # loss_sigma += loss.item()
        # # plccTotal += rho_p
        # print('plcc:%f' % rho_p)
        # print('srocc:%f'% rho_s)
        # print("[Training]: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} PLCC:{:.4f}".format(
        #     epoch + 1, config.n_epoch, it + 1, len(train_loader), loss.item(), plccTotal/(it+1)))

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    # save weights
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    if (epoch + 1) % config.save_freq == 0:
        weights_file_name = time_str + "_epoch%d.pth" % (epoch + 1)
        weights_file = os.path.join(config.weights_path, weights_file_name)
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