import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import random
import os 
import numpy as np 
import PIL
from PIL import Image
import pdb 

import torch
from torch import autocast

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imagenet_labels import ind2name
from typing import Any, Callable, Dict, List, Optional, Union
from editor import ImageEditor
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Diffusion based data generator')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')

parser.add_argument('--start_seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--end_seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--method', type=str, default='none',
                    help='method for data generation (none, positive, negative, prompt)')
parser.add_argument('--noise', default=0.5, type=float, help='noise rate for forward diffusion')
parser.add_argument('--nway', default=10, type=int, help='n-way classification')
parser.add_argument('--budget', default=100, type=int, help='max generated samples per class')
parser.add_argument('--kshot', default=5, type=int, help='n-shot classification')
parser.add_argument('--output_dir', type=str, default='./data/',
                    help='path to store the generated data')
parser.add_argument('--confusion_matrix', type=str, default='./data/',
                    help='path to confusion matrix')
parser.add_argument('--prompt', type=str, default='',
                    help='path to confusion matrix')

parser.add_argument('--imagenet', action='store_true', help="generate from imagenet")

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")

best_acc1 = 0



def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = parser.parse_args()

    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
        
    main_worker(args.gpu, ngpus_per_node, args)

    
  


def main_worker(gpu, ngpus_per_node, args):
    
    
    editor = ImageEditor()
    
    total_seeds = args.end_seed - args.start_seed 
    
    for sind, seed in enumerate(range(args.start_seed, args.end_seed)):
        
        print("processing seed {}, {}/{}".format(seed, sind+1, total_seeds))

        
        set_all_seeds(seed)
        
        dataset_path = args.output_dir
        dataset_path = os.path.join(dataset_path , 'dataset{}'.format(seed))
        src_data = os.path.join(args.data , 'dataset{}/train/'.format(seed))
        
        dataset = datasets.ImageFolder(
            src_data,
            transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
            ]))
    
        cls_path = [f.path for f in os.scandir(src_data) if f.is_dir()]
        cls_path = [c.split("/")[-1] for c in cls_path]
        cls_path = np.sort(cls_path)
        cls_names = [c[10:] for c in cls_path]

        if 'txt2img' in args.method: 
            train_path = os.path.join(dataset_path, 'train_{}'.format(args.method))
        else:
            train_path = os.path.join(dataset_path, 'train_{}_noise{}'.format(args.method, args.noise))
        os.makedirs(train_path, exist_ok = True) 

        t = transforms.Compose([transforms.Resize(512),transforms.CenterCrop(512),])


        for ind, cls in enumerate(cls_names): 

            class_path = os.path.join(train_path, cls_path[ind])
            os.makedirs(class_path, exist_ok = True)

            if 'genie' in args.method: 
                imgs = [c[0] for c in dataset.samples if c[1]!=ind]
            else:
                imgs = [c[0] for c in dataset.samples if c[1]==ind]

            print("processing class {}, {}/{}".format(cls, ind+1 , len(cls_names)))

            for k,img in enumerate(imgs): 

                if 'genie' in args.method: 
                    prompt = 'a photo of a {}.'.format(cls)

                    img_ = Image.open(img).convert('RGB')
                    img_ = t(img_)

                    output = editor.edit(img_, prompt, args.noise)
                    img_path = os.path.join(class_path, '{}_genie.jpg'.format(k))
                    output.save(img_path)

                if 'img2img' in args.method:

                    prompt = 'a photo of a {}.'.format(cls)
                    img_ = Image.open(img).convert('RGB')
                    img_ = t(img_)

                    for j in range(len(cls_names)-1):
                        output = editor.edit(img_, prompt, args.noise)
                        img_path = os.path.join(class_path, '{}img2img{}.jpg'.format(k,j))
                        output.save(img_path)

                if 'txt2img' in args.method:
                    prompt = 'a photo of a {}.'.format(cls)
                    img_ = Image.open(img).convert('RGB')
                    img_ = t(img_)

                    for j in range(len(cls_names)-1):
                        output = editor.edit(img_, prompt, args.noise)
                        img_path = os.path.join(class_path, '{}_txt2img{}.jpg'.format(k,j))
                        output.save(img_path)








if __name__ == '__main__':
    main()