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


# @torch.no_grad()
def evaluate_fewshot_v2(
    encoder, transform, caching_epochs, augs_name, save_path, data_path="/home/datadrive/mini_imagenet_fs", n_way=5, n_shots=1, n_query=16, classifier='LR', power_norm=False, fine_tuning_epochs=None):

    encoder.eval()
    
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
        if augs_name:
            if os.path.exists(os.path.join(dataset_path, episode, augs_name)):
                episode_support_diffaugs = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, episode, augs_name), transform=transform)
            else:
                dataset_path_diffaugs = os.path.join("/home/datadrive/mini_imagenet_fs_ablation", '5way_{}shot'.format(n_shots))
                episode_support_diffaugs = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path_diffaugs, episode, augs_name), transform=transform)
            support_dataloader_diffaugs = DataLoader(episode_support_diffaugs, batch_size=16, shuffle=True, num_workers=4)

        support_x = []; support_y = []
        query_x = []; query_y = []

        for _ in range(caching_epochs):
            for idx, (images, labels) in enumerate(support_dataloader):
                # breakpoint()
                images = images.cuda(non_blocking=True)
                f = encoder(images)
                f = f/f.norm(dim=-1, keepdim=True)
                if power_norm:
                    f = f ** 0.5
                if "Linear" in classifier:
                    support_x.append(f)
                    support_y.append(labels)
                else:
                    support_x.append(f.detach().cpu().numpy())
                    support_y.append(labels.detach().cpu().numpy())
            
        for idx, (images, labels) in enumerate(query_dataloader):
                images = images.cuda(non_blocking=True)
                f = encoder(images)
                f = f/f.norm(dim=-1, keepdim=True)
                if power_norm:
                    f = f ** 0.5
                if "Linear" in classifier:
                    query_x.append(f)
                    query_y.append(labels)
                else:
                    query_x.append(f.detach().cpu().numpy())
                    query_y.append(labels.detach().cpu().numpy())

        if augs_name:
            for idx, (images, labels) in enumerate(support_dataloader_diffaugs):
                    # breakpoint()
                    images = images.cuda(non_blocking=True)
                    f = encoder(images)
                    f = f/f.norm(dim=-1, keepdim=True)
                    if power_norm:
                        f = f ** 0.5
                    if "Linear" in classifier:
                        support_x.append(f)
                        support_y.append(labels)
                    else:
                        support_x.append(f.detach().cpu().numpy())
                        support_y.append(labels.detach().cpu().numpy())

        if classifier == 'LR':
            clf = LogisticRegression(penalty='l2',
                                    random_state=0,
                                    C=1.0,
                                    solver='lbfgs',
                                    max_iter=1000,
                                    multi_class='multinomial')
            clf.fit(np.concatenate(support_x), np.concatenate(support_y))
            cur_qry_pred = clf.predict(np.concatenate(query_x))
            acc = metrics.accuracy_score(np.concatenate(query_y), cur_qry_pred)

        elif classifier == 'SVM':
            clf = LinearSVC(C=1.0)
            clf.fit(np.concatenate(support_x), np.concatenate(support_y))
            cur_qry_pred = clf.predict(np.concatenate(query_x))
            acc = metrics.accuracy_score(np.concatenate(query_y), cur_qry_pred)
        
        elif classifier == 'Linear':
            # breakpoint()
            linear_clf = nn.Linear(in_features=f.shape[-1], out_features=n_way)
            linear_clf = linear_clf.cuda()

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
            
            torch.save(linear_clf, os.path.join(save_path, "{}_linear_ckpt.pt".format(episode)))    
            scores = linear_clf(query_x)
            acc = metrics.accuracy_score(query_y, scores.argmax(dim=1).cpu())
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
            acc = metrics.accuracy_score(query_y, scores.argmax(dim=1).cpu())
            #breakpoint()
        
        print(acc)
        accs.append(acc)

    
    mean = np.array(accs).mean()
    std = np.array(accs).std()
    c95 = 1.96*std/math.sqrt(np.array(accs).shape[0])
    print('classifier: {}, power_norm: {}, diffusion_augs: {}, {}-way {}-shot acc: {:.2f}+{:.2f}'.format(
        classifier, power_norm, augs_name, n_way, n_shots, mean*100, c95*100))
    