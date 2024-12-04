import argparse
import os
import random
import time
from math import floor
import ipdb

import h5py
import numpy as np
import openslide
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

from tadgraph.embedder.ctrans import ctranspath, trnsfrms_val
from tadgraph.datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from tadgraph.models.backbones.resnet_custom import resnet50_baseline
from tadgraph.models.backbones.hipt_4k import HIPT_4K
from tadgraph.models.backbones.hipt_model_utils import eval_transforms
from tadgraph.utils.file_utils import save_hdf5
from tadgraph.utils.utils import collate_features, print_network
from tadgraph.paths import *

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class fully_connected(nn.Module):
    """docstring for BottleNeck"""

    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return out_1, out_3


def compute_w_loader(output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, custom_transforms=None, target_patch_size=-1):
    """
    args:
            output_path: directory to save computed features (.h5 file)
            model: pytorch model
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            pretrained: use weights pretrained on imagenet
            custom_downsample: custom defined downscale factor of image patches
            target_patch_size: custom defined, rescaled image size before embedding
    """

    downscale = 64
    vis_level = wsi.get_best_level_for_downsample(downscale)
    w, h = wsi.level_dimensions[vis_level]
    downsamples = wsi.level_downsamples[vis_level]
    whole_wsi = wsi.read_region(
        (0, 0), level=vis_level, size=(w, h)).convert("RGB")
    whole_wsi = whole_wsi.resize((target_patch_size, target_patch_size))
    img = custom_transforms(whole_wsi)
    img = img.unsqueeze(0)
    img = img.to(device, non_blocking=True)

    features, _ = model(img)
    torch.save(features.cpu(), output_path)

    return output_path


'''
CUDA_VISIBLE_DEVICES=0 python extract_patches_for_thunmnail.py --task tcga_rcc
'''

parser = argparse.ArgumentParser(
    description='Feature Extraction for thumbnail')
parser.add_argument('--task', type=str, default='tcga_brca',
                    choices=['camelyon16', 'tcga_nsclc', 'tcga_rcc', 'tcga_brca',
                             'tcga_blca', 'tcga_ucec', 'tcga_esca', 'tcga_prad'])
parser.add_argument('--model', type=str, default='kimianet',
                    choices=['resnet_50', 'hipt', 'kimianet', 'ctrans'])
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=512)
args = parser.parse_args()


if __name__ == '__main__':
    args.data_h5_dir = "thumbnail"
    if args.task == "camelyon16":
        args.data_h5_dir = os.path.join(CAMELYON16_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(CAMELYON16_DATA_DIR, "WSIs")
        args.slide_ext = ".tif"
    elif args.task == "tcga_nsclc":
        args.data_h5_dir = os.path.join(NSCLC_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(NSCLC_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "tcga_rcc":
        args.data_h5_dir = os.path.join(RCC_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(RCC_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "tcga_brca":
        args.data_h5_dir = os.path.join(BRCA_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(BRCA_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "tcga_blca":
        args.data_h5_dir = os.path.join(BLCA_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(BLCA_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "tcga_ucec":
        args.data_h5_dir = os.path.join(UCEC_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(UCEC_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "tcga_esca":
        args.data_h5_dir = os.path.join(ESCA_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(ESCA_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "tcga_prad":
        args.data_h5_dir = os.path.join(PRAD_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(PRAD_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"

    # define feature directory
    if args.model == "resnet_50":
        args.feat_dir = os.path.join(
            args.data_h5_dir, "resnet50_trunc_pt_patch_features")
    elif args.model == "hipt":
        args.feat_dir = os.path.join(
            args.data_h5_dir, "vits_tcga_pancancer_dino_pt_patch_features")
    elif args.model == "kimianet":
        args.feat_dir = os.path.join(
            args.data_h5_dir, "kimianet_pt_patch_features")
    elif args.model == "ctrans":
        args.feat_dir = os.path.join(
            args.data_h5_dir, "ctrans_pt_patch_features")

    print('initializing dataset')

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    print(f'loading model checkpoint: {args.model}')
    if args.model == "resnet_50":
        model = resnet50_baseline(pretrained=True)
        args.custom_transforms = None
        model = model.to(device)
        model = nn.DataParallel(model)
    elif args.model == "hipt":
        model = HIPT_4K()
        args.custom_transforms = eval_transforms()
        model = model.to(device)
        model = nn.DataParallel(model)
    elif args.model == "kimianet":
        densenet_model = torchvision.models.densenet121(pretrained=True)
        densenet_model.features = nn.Sequential(
            densenet_model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        num_ftrs = densenet_model.classifier.in_features
        model = fully_connected(densenet_model.features, num_ftrs, 30)
        args.custom_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # freeze model
        for param in model.parameters():
            param.requires_grad = False
        model = model.to(device)
        model = nn.DataParallel(model)
        # load pretrained weights
        # model.load_state_dict(torch.load(
        #     "/home/r20user2/Documents/MIL_Interpretation/data/KimiaNet_weights/KimiaNetPyTorchWeights.pth"))
        model.load_state_dict(torch.load(os.path.join(
            KIMIANET_WEIGHT, "KimiaNetPyTorchWeights.pth")))
    elif args.model == "ctrans":
        model = ctranspath()
        model.head = nn.Identity()
        args.custom_transforms = trnsfrms_val
        # Load pretrained checkpoints
        state_dict = torch.load('./pretrained_models/ctranspath.pth',
                                map_location="cpu")["model"]
        model.load_state_dict(state_dict, strict=True)
        # state_dict = clean_state_dict_clip(state_dict)
        # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        # print('missing keys: ', missing_keys)
        # print('unexpected keys: ', unexpected_keys)
        # import ipdb
        # ipdb.set_trace()
        model = model.to(device)
        model = nn.DataParallel(model)

    model.eval()
    print_network(model)

    model.eval()

    wsi_files = os.listdir(args.data_slide_dir)
    total = len(wsi_files)

    for bag_candidate_idx in range(total):
        slide_id = wsi_files[bag_candidate_idx].split(args.slide_ext)[0]
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        slide_file_path = os.path.join(
            args.data_slide_dir, slide_id+args.slide_ext)

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_path = os.path.join(args.feat_dir, "pt_files", slide_id+'.pt')
        output_file_path = compute_w_loader(output_path, wsi,
                                            model=model, batch_size=1, verbose=1,
                                            custom_transforms=args.custom_transforms,
                                            target_patch_size=args.target_patch_size)
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(
            output_file_path, time_elapsed))
