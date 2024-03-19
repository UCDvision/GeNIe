# GeNIe: Generative Hard Negative Images Through Diffusion

This Repository is an official implementation of [GeNIe](https://arxiv.org/abs/2312.02548).
Our code is based on [BOOMERANG](https://colab.research.google.com/drive/1PV5Z6b14HYZNx1lHCaEVhId-Y4baKXwt). 

## Overview


GeNIe is a novel data augmentation technique employing Generative text-based latent Diffusion models. GeNIe merges contrasting data points (an image from the source category and a text prompt from the target category) using a latent diffusion model conditioned on the text prompt. By limiting diffusion iterations, we preserve low-level and background features from the source image while representing the target category, creating challenging samples. Additionally, our method further enhances its effectiveness by dynamically adjusting the noise level for each image, known as GeNIe-Ada. Rigorous experimentation across both few-shot and long-tail distribution scenarios substantiates the superior performance of our approach over existing techniques.

![genie_teaser4](https://github.com/UCDvision/GeNIe/assets/62820830/33aea37e-cfaa-4f5e-824a-cd7d729b451c)

## Requirements

All our experiments use the PyTorch library. Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). We used Python 3.7 for our experiments.

## Getting Started 
Please install dependencies in a virtual environment: 
 
 ```
 pip install transformers
 pip install diffusers==0.19.0
 ```
## Demo

You can try GeNIe with this colab [GeNIe Colab](https://colab.research.google.com/drive/1Q3jBK4sfyNU5b1dQZgIP8uS2ObzbIE8X). 



## Few-Shot on tiered-ImageNet and mini-ImageNet


1. Use the imagenet_sampler.ipynb notebook to generate episodes for mini-ImageNet and tiered-ImageNet

2. To generate augmentations for all baselines (txt2img, img2img, and GeNIe), utilize the generate_data.sh script as follows:
```
generate_data.sh 0 20 30
```
This script employs GPU 0 to generate augmentations for episodes 20 to 30.

3. To generate Noise Adaptive dataset for few-shot learning - run `/few_shot/noise_adaptive.ipynb`

3. Train on augmented dataset:
```
CUDA_VISIBLE_DEVICES=0 python ./train.py --data_path /home/datadrive/mini_imagenet_fs --backbone resnet18 --eval_path /home/datadrive/mini_imagenet_fs/models/mini_r18_v2.pth --transform weak --caching_epochs 5 --n_shots 1 --clf LR --augs_name train_negative_noise_noad_r18_v3 &> PATH_TO_LOG.txt & 
```
Change all paths and GPU/CUDA device IDs as per your settings. 

## Citation

If you make use of the code, please cite the following work:
```
@misc{koohpayegani2023genie,
      title={GeNIe: Generative Hard Negative Images Through Diffusion}, 
      author={Soroush Abbasi Koohpayegani and Anuj Singh and K L Navaneet and Hadi Jamali-Rad and Hamed Pirsiavash},
      year={2023},
      eprint={2312.02548},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is under the MIT license.
