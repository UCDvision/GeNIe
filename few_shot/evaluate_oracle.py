import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
from torchvision import transforms

import numpy as np
import math

from mapping import map

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# @torch.no_grad()
def evaluate_fewshot_oracle(
    encoder, transform, caching_epochs, augs_name, data_path="/home/datadrive/mini_imagenet_fs", n_way=5, n_shots=1, power_norm=False):

    encoder.eval()
    
    dataset_path = os.path.join(data_path, '5way_{}shot'.format(n_shots)) 
    episode_datasets = os.listdir(dataset_path)
    

    accs = []
    for episode in tqdm(episode_datasets):
        episode_support = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, episode, "train"), transform=transform)

        support_dataloader = DataLoader(episode_support, batch_size=16, shuffle=True, num_workers=4)

        ## Our Augmentations
        if augs_name:
            if os.path.exists(os.path.join(dataset_path, episode, augs_name)):
                episode_support_diffaugs = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, episode, augs_name), transform=transform)
            else:
                dataset_path_diffaugs = os.path.join("/home/datadrive/mini_imagenet_fs_ablation", '5way_{}shot'.format(n_shots))
                episode_support_diffaugs = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path_diffaugs, episode, augs_name), transform=transform)
            support_dataloader_diffaugs = DataLoader(episode_support_diffaugs, batch_size=16, shuffle=True, num_workers=4)

        support_x = []; support_y = []

        for _ in range(caching_epochs):
            for idx, (images, labels) in enumerate(support_dataloader):
                # breakpoint()
                images = images.cuda(non_blocking=True)
                f = encoder(images)
                f = f/f.norm(dim=-1, keepdim=True)
                if power_norm:
                    f = f ** 0.5
                support_x.append(f.detach().cpu().numpy())
                support_y.append(labels.detach().cpu().numpy())

        if augs_name:
            for idx, (images, labels) in enumerate(support_dataloader_diffaugs):
                    # breakpoint()
                    images = images.cuda(non_blocking=True)
                    f = encoder(images)
                    f = f/f.norm(dim=-1, keepdim=True)
                    if power_norm:
                        f = f ** 0.5
                    support_x.append(f.detach().cpu().numpy())
                    support_y.append(labels.detach().cpu().numpy())

        cls_names = os.listdir(os.path.join(dataset_path, episode, "train"))
        cls_names = np.sort(cls_names)
        cls_names = [c.split("_")[-1] for c in cls_names]
        cls_ids = [list(map.values()).index(cls) for cls in cls_names]

        support_x = np.concatenate(support_x)
        support_y = np.concatenate(support_y)

        acc = metrics.accuracy_score(support_y, support_x[:, cls_ids].argmax(axis=-1))


        print(acc)
        accs.append(acc)

    
    mean = np.array(accs).mean()
    std = np.array(accs).std()
    c95 = 1.96*std/math.sqrt(np.array(accs).shape[0])
    print('DEIT, power_norm: {}, diffusion_augs: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
        power_norm, augs_name, n_way, n_shots, mean*100, c95*100))
    