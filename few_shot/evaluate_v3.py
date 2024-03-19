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

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def one_hot(labels, num_classes):
    labels = labels.reshape(labels.shape[0], 1)
    one_hot_target = (labels == torch.arange(num_classes).reshape(1, num_classes)).float()
    return one_hot_target

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# @torch.no_grad()
def evaluate_fewshot_v3(
    encoder, transform, caching_epochs, diff_augs_name, det_aug_name, data_path="/home/datadrive/mini_imagenet_fs", n_way=5, n_shots=1, n_query=16, classifier='Linear', power_norm=False, fine_tuning_epochs=None):

    encoder.eval()
    encoder = encoder.cpu()
    
    dataset_path = os.path.join(data_path, '5way_{}shot'.format(n_shots)) 
    episode_datasets = os.listdir(dataset_path)
    
    query_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    accs = []
    for episode in tqdm(episode_datasets):
        episode_support = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, episode, "train"), transform=transform)
        episode_query = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, episode, "val"), transform=query_transform)

        support_dataloader = DataLoader(episode_support, batch_size=16, shuffle=True, num_workers=4)
        query_dataloader = DataLoader(episode_query, batch_size=16, shuffle=False, num_workers=4)

        ## Our Augmentations
        if diff_augs_name:
            episode_support_diffaugs = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, episode, diff_augs_name), transform=transform)
            support_dataloader_diffaugs = DataLoader(episode_support_diffaugs, batch_size=16, shuffle=True, num_workers=4)

        support_x = []; support_y = []
        query_x = []; query_y = []

        for _ in range(caching_epochs):
            for idx, (images, labels) in enumerate(support_dataloader):
                images = images.cpu()#.cuda(non_blocking=True)
                f = encoder(images)
                f = f/f.norm(dim=-1, keepdim=True)
                if power_norm:
                    f = f ** 0.5
                support_x.append(f)
                labels = one_hot(labels, num_classes=n_way)
                support_y.append(labels)

            
        for idx, (images, labels) in enumerate(query_dataloader):
                images = images.cpu()#.cuda(non_blocking=True)
                f = encoder(images)
                f = f/f.norm(dim=-1, keepdim=True)
                if power_norm:
                    f = f ** 0.5
                query_x.append(f)
                labels = one_hot(labels, num_classes=n_way)
                query_y.append(labels)

        if diff_augs_name:
            for idx, (images, labels) in enumerate(support_dataloader_diffaugs):
                    images = images.cpu()#.cuda(non_blocking=True)
                    f = encoder(images)
                    f = f/f.norm(dim=-1, keepdim=True)
                    if power_norm:
                        f = f ** 0.5
                    support_x.append(f)
                    labels = one_hot(labels, num_classes=n_way)
                    support_y.append(labels)
                    

        for _ in range(caching_epochs):
            if 'mixup' in det_aug_name:
                for (x1, y1), (x2, y2) in zip(support_dataloader, support_dataloader):
                    y1 = one_hot(y1, num_classes=n_way)
                    y2 = one_hot(y2, num_classes=n_way)

                    x1 = x1.cpu()#.cuda(non_blocking=True)
                    x2 = x2.cpu()#.cuda(non_blocking=True)
                    # y1 = y1.cuda(non_blocking=True); y2 = y2.cuda(non_blocking=True)
                    lam = np.random.beta(0.8, 0.8)
                    x = Variable(lam * x1 + (1. - lam) * x2)
                    y = Variable(lam * y1 + (1. - lam) * y2)
                    f = encoder(x)
                    f = f/f.norm(dim=-1, keepdim=True)
                    if power_norm:
                        f = f ** 0.5
                    support_x.append(f)
                    support_y.append(y)

            if 'cutmix' in det_aug_name:
                #support_x_cutmix = []; support_ya_cutmix = []; support_yb_cutmix = []
                for idx, (images, labels) in enumerate(support_dataloader):
                    images = images.cpu()#.cuda(non_blocking=True)
                    labels = one_hot(labels, num_classes=n_way)
                    #labels = labels.cuda()
                    # r = np.random.rand(1)
                    beta = 0.8
                    # generate mixed sample
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(images.size()[0])#.cuda()
                    target_a = labels
                    target_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    # generate mxed labels
                    target = target_a * lam + target_b * (1. - lam)
                    # compute output
                    f = encoder(images)
                    f = f/f.norm(dim=-1, keepdim=True)
                    if power_norm:
                        f = f ** 0.5
                    support_x.append(f)
                    support_y.append(target)
                    
                    #loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                
                
                
        
        if classifier == 'Linear':
            # breakpoint()
            linear_clf = nn.Linear(in_features=f.shape[-1], out_features=n_way)
            linear_clf = linear_clf.cuda()

            optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.1, momentum=0.9, dampening=0.9, weight_decay=0.001)

            loss_function = nn.CrossEntropyLoss()
            loss_function = loss_function.cuda()

            support_x = torch.concat(support_x).cuda()
            query_x = torch.concat(query_x).cuda()
            #breakpoint()
            support_y = torch.concat(support_y).cuda()
            query_y = torch.concat(query_y).cuda()

            #breakpoint()
            support_size = support_x.shape[0]
            batch_size = 16 # support_size
            # print("Fine-tuning")
            for epoch in range(fine_tuning_epochs):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size , batch_size):
                    optimizer.zero_grad()
                    selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                    x_batch = support_x[selected_id]
                    y_batch = support_y[selected_id] 
                    scores = linear_clf(x_batch.detach())
                    # loss = scores.pow(2).sum()
                    loss = loss_function(scores, y_batch)
                    # loss = torch.autograd.Variable(loss, requires_grad=True)
                    loss.backward()
                    optimizer.step()
                    #breakpoint()
            
    
            scores = linear_clf(query_x)
            acc = metrics.accuracy_score(query_y.argmax(dim=1).cpu(), scores.argmax(dim=1).cpu())
            #breakpoint()

        elif classifier == 'protoLinear':
            # breakpoint()
            class distLinear(nn.Module):
                def __init__(self, indim, outdim):
                    super(distLinear, self).__init__()
                    self.L = nn.Linear(in_features=indim, out_features=outdim, bias=False)
                    self.L = self.L.cuda()
                    self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
                    if self.class_wise_learnable_norm:      
                        WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

                    if outdim <=200:
                        self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
                    else:
                        self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

                def forward(self, x):
                    # breakpoint()
                    x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
                    x_normalized = x.div(x_norm+ 0.00001)
                    if not self.class_wise_learnable_norm:
                        L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
                        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
                    cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
                    scores = self.scale_factor* (cos_dist) 

                    return scores

            linear_clf = distLinear(indim=f.shape[-1], outdim=n_way)
            optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.1, momentum=0.9, dampening=0.9, weight_decay=0.001)

            loss_function = nn.CrossEntropyLoss()
            loss_function = loss_function.cuda()

            support_x = torch.concat(support_x)
            query_x = torch.concat(query_x)
            support_y = torch.concat(support_y).long().cuda()
            query_y = torch.concat(query_y)

            #breakpoint()
            support_size = support_x.shape[0]
            batch_size = 16 # support_size
            # print("Fine-tuning")
            for epoch in range(fine_tuning_epochs):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size , batch_size):
                    optimizer.zero_grad()
                    selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
                    x_batch = support_x[selected_id]
                    y_batch = support_y[selected_id] 
                    scores = linear_clf(x_batch.detach())
                    # loss = scores.pow(2).sum()
                    loss = loss_function(scores, y_batch)
                    # loss = torch.autograd.Variable(loss, requires_grad=True)
                    loss.backward()
                    optimizer.step()
                    #breakpoint()
            
            # breakpoint()
            scores = linear_clf(query_x)
            acc = metrics.accuracy_score(query_y.argmax(dim=1).cpu(), scores.argmax(dim=1).cpu())
            #breakpoint()
        print(acc)
        accs.append(acc)

    
    mean = np.array(accs).mean()
    std = np.array(accs).std()
    c95 = 1.96*std/math.sqrt(np.array(accs).shape[0])
    print('classifier: {}, power_norm: {}, diffusion_augs: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
        classifier, power_norm, diff_augs_name, n_way, n_shots, mean*100, c95*100))
    