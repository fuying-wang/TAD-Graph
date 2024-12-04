import argparse
import os
import time
import ipdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tadgraph.datasets.patch_dataset import Whole_Slide_Bag
from tadgraph.models.backbones.cnn_backbone import resnet_18, resnet_50
from tadgraph.models.backbones.resnet_custom import resnet50_baseline
from tadgraph.paths import *

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
commands:
CUDA_VISIBLE_DEVICES=1,2 python extract_features.py --task camelyon16 --wsi_patch_dir extracted_mag20x_patch224/patches --feat_dir extracted_mag20x_patch224/resnet50_custom_pretrained_pt_features_v2
'''

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument("--task", type=str, default="camelyon16", choices=["camelyon16", "tcga_nsclc", "tcga_rcc"])
parser.add_argument('--wsi_patch_dir', type=str,
                    default='extracted_mag20x_patch224/img_patches')
parser.add_argument('--batch_size', type=int, default=1536)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--feat_dir', type=str,
                    default='extracted_mag20x_patch224/resnet50_custom_pretrained_pt_features')
parser.add_argument('--model', type=str, default="resnet_50")
parser.add_argument('--num_workers', type=int, default=16)
args = parser.parse_args()


def extract_patch_features_per_slide(slide_id, feat_dir, batch_size=8, num_workers=16, print_every=5):
    print(f"processing {slide_id}")
    bag_dir = os.path.join(args.wsi_patch_dir, slide_id)
    wsi_dataset = Whole_Slide_Bag(bag_dir, pretrained=True)

    loader = DataLoader(dataset=wsi_dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True)

    features_list = []
    with torch.no_grad():
        for idx, imgs in enumerate(loader):
            if idx % print_every == 0:
                print(
                    f"batch {idx}/{len(loader)}, {idx * batch_size} files processed")
            imgs = imgs.to(device, non_blocking=True)
            features = model(imgs)
            features = features.detach().cpu()
            features_list.append(features)

    print(f"processing {slide_id} done\n")
    save_file_path = os.path.join(feat_dir, slide_id + ".pt")
    slide_features = torch.cat(features_list, dim=0)
    torch.save(slide_features, save_file_path)


if __name__ == "__main__":
    # define the model
    # these are original cnn models pretrained on ImageNet
    # if args.model == "resnet_18":
    #     model, _ = resnet_18(pretrained=True)
    # elif args.model == "resnet_50":
    #     model, _ = resnet_50(pretrained=True)
    # else:
    #     raise ValueError(f"no model named {args.model}")

    if args.task.lower() == "camelyon16":
        args.wsi_patch_dir = os.path.join(CAMELYON16_DATA_DIR, args.wsi_patch_dir)
        args.feat_dir = os.path.join(CAMELYON16_DATA_DIR, args.feat_dir)
    elif args.task.lower() == "tcga_nsclc": 
        args.wsi_patch_dir = os.path.join(NSCLC_DATA_DIR, args.wsi_patch_dir)
        args.feat_dir = os.path.join(NSCLC_DATA_DIR, args.feat_dir)
    elif args.task.lower() == "tcga_rcc":
        args.wsi_patch_dir = os.path.join(RCC_DATA_DIR, args.wsi_patch_dir)
        args.feat_dir = os.path.join(RCC_DATA_DIR, args.feat_dir)

    wsi_folders = sorted(os.listdir(args.wsi_patch_dir))

    os.makedirs(args.feat_dir, exist_ok=True)
    dest_files = os.listdir(args.feat_dir)

    if args.model == "resnet_50":
        model = resnet50_baseline(pretrained=True)

    model = model.to(device)
    print(f"Loaded {args.model} checkpoint")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    total = len(wsi_folders)

    for bag_candidate_idx in range(total):

        slide_id = wsi_folders[bag_candidate_idx]
        time_start = time.time()

        if not args.no_auto_skip and slide_id+".pt" in dest_files:
            print(f"skipped {slide_id}")
            continue

        extract_patch_features_per_slide(
            slide_id, args.feat_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        time_elapsed = time.time() - time_start
        print(
            f"\ncomputing features for {slide_id} took {time_elapsed} s")
        print(f"\nprogress: {bag_candidate_idx}/{total}")
        print(slide_id)