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


def compute_w_loader(file_path, output_path, wsi, model,
                     batch_size=8, verbose=0, print_every=20, pretrained=True,
                     custom_downsample=1, custom_transforms=None, target_patch_size=-1):
    """
    args:
            file_path: directory of bag (.h5 file)
            output_path: directory to save computed features (.h5 file)
            model: pytorch model
            batch_size: batch_size for computing features in batches
            verbose: level of feedback
            pretrained: use weights pretrained on imagenet
            custom_downsample: custom defined downscale factor of image patches
            target_patch_size: custom defined, rescaled image size before embedding
    """

    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained,
                                 custom_downsample=custom_downsample, custom_transforms=custom_transforms,
                                 target_patch_size=target_patch_size)
    x, y = dataset[0]
    kwargs = {'num_workers': 16,
              'pin_memory': True} if device.type == "cuda" else {}
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        **kwargs, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    mode = 'w'
    try:
        for count, (batch, coords) in enumerate(loader):
            with torch.no_grad():
                if count % print_every == 0:
                    print('batch {}/{}, {} files processed'.format(count,
                        len(loader), count * batch_size))
                batch = batch.to(device, non_blocking=True)
                if args.model == "hipt":
                    features = model(batch)
                elif args.model == "resnet_50":
                    features = model(batch)
                elif args.model == "kimianet":
                    features, _ = model(batch)
                elif args.model == "ctrans":
                    features = model(batch)
                elif args.model == "uni":
                    features = model(batch)
                else:
                    raise NotImplementedError

                features = features.cpu().numpy()

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
    except Exception as e:
        print('error in processing {}: {}'.format(file_path, e))
        return None

    return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--task', type=str, default='camelyon16',
                    choices=['camelyon16', 'tcga_nsclc', 'tcga_rcc', 'tcga_brca', 
                             'tcga_blca', 'tcga_ucec', 'tcga_esca', 'tcga_prad',
                             'tcga_read', 'panda', 'ebrains', 'tcga_coad',
                             'cptac_rcc', 'cptac_brca'
                             ])
parser.add_argument('--model', type=str, default='kimianet',
                    choices=['resnet_50', 'hipt', 'kimianet', 'ctrans', 'uni'])
parser.add_argument('--data_h5_dir', type=str,
                    default="extracted_mag20x_patch512")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=256)
args = parser.parse_args()


if __name__ == '__main__':
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
    elif args.task == "tcga_read":
        args.data_h5_dir = os.path.join(READ_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(READ_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "panda":
        args.data_h5_dir = os.path.join(PANDA_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(PANDA_DATA_DIR, "WSIs")
        args.slide_ext = ".tiff"
    elif args.task == "ebrains":
        args.data_h5_dir = os.path.join(EBRAINS_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(EBRAINS_DATA_DIR, "WSIs")
        args.slide_ext = ".ndpi"
    elif args.task == "tcga_coad":
        args.data_h5_dir = os.path.join(COAD_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(COAD_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "cptac_rcc":
        args.data_h5_dir = os.path.join(CPTAC_RCC_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(CPTAC_RCC_DATA_DIR, "WSIs")
        args.slide_ext = ".svs"
    elif args.task == "cptac_brca":
        args.data_h5_dir = os.path.join(CPTAC_BRCA_DATA_DIR, args.data_h5_dir)
        args.data_slide_dir = os.path.join(CPTAC_BRCA_DATA_DIR, "WSIs")
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
    elif args.model == "uni":
        args.feat_dir = os.path.join(
            args.data_h5_dir, "uni_pt_patch_features")
    else:
        raise NotImplementedError
        
    args.csv_path = os.path.join(args.data_h5_dir, "process_list_autogen.csv")

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    print(f'loading model checkpoint: {args.model}')
    if args.model == "resnet_50":
        model = resnet50_baseline(pretrained=True)
        args.custom_transforms = None
        model = model.to(device)
        model = nn.DataParallel(model)
    elif args.model == "hipt":
        model = HIPT_4K(
            model256_path=str(HIPT_DIR / "vit256_small_dino.pth"),
            model4k_path=str(HIPT_DIR / "vit4k_xs_dino.pth"),
        )
        args.custom_transforms = eval_transforms(imagesize=args.target_patch_size)
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
        model.load_state_dict(torch.load(os.path.join(
            KIMIANET_WEIGHT, "KimiaNetPyTorchWeights.pth")))
    elif args.model == "ctrans":
        model = ctranspath()
        model.head = nn.Identity()
        args.custom_transforms = trnsfrms_val
        # Load pretrained checkpoints
        state_dict = torch.load('../../pretrained/CTransPath/ctranspath.pth', 
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
    elif args.model == "uni":
        import timm
        local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        if os.path.exists(os.path.join(local_dir, "pytorch_model.bin")):
            print("Model already downloaded.")
        else:
            # Load UNI to extract patch features
            from huggingface_hub import login, hf_hub_download
            login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
            hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        args.custom_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        model = model.to(device)
        model = nn.DataParallel(model)

    model.eval()
    print_network(model)

    model.eval()
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(
            args.data_slide_dir, slide_id+args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        if not os.path.exists(h5_file_path):
            print(f"{h5_file_path} doesn't exist!")
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
                                            model=model, batch_size=args.batch_size, verbose=1, print_every=20,
                                            custom_downsample=args.custom_downsample,
                                            custom_transforms=args.custom_transforms,
                                            target_patch_size=args.target_patch_size)
        if output_file_path is None:
            print('error in processing {}'.format(bag_name))
            continue
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(
            output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(
            args.feat_dir, 'pt_files', bag_base+'.pt'))
