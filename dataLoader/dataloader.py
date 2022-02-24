from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from itertools import chain
from glob import glob
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import os
import cv2
import torch
import math


class AVADataset(Dataset):

    def __init__(self, transforms = None, train=True, test=False, txtFileName=''):
        self.train = train
        self.test = test
        path = "./temp/AVA_scores.txt"
        imagePath = ""
        if self.test:
            imagePath = "./dataset/test"
        else:
            imagePath = "./dataset/train"
        labels = []
        imageData = []
        filename = open(path, mode="r", encoding="utf-8")
        f = filename.read().split("\n")
        for fn in tqdm(f):
            data = fn.split()
            if len(data) == 0:
                break
            p = os.path.join(imagePath, data[0]+".jpg")
            # p = imagePath+"/"+data[0]+".jpg"
            label = data[1]
            if os.path.exists(p):
                imageData.append((p, label))
        self.imageData = imageData
        self.labels = labels
        if transforms is None:
            self.transforms = T.Compose([
                T.ToTensor(),
                ])
    def __getitem__(self, index):
        if self.test:
            filename, label = self.imageData[index]
            img = Image.open(filename).convert('RGB')
            img = self.transforms(img)
            return img, float(label)
        else:
            filename, label = self.imageData[index]
            img = Image.open(filename).convert('RGB')
            img = self.transforms(img)
            label = float(label)
            return img, label
    def __len__(self):
        return len(self.imageData)
if __name__=="__main__":
    c = AVADataset()

    for i in range(5000):
        print(c[i])