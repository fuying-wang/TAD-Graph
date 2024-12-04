import argparse
import os
import ipdb

import numpy as np
import pandas as pd
from tadgraph.datasets.dataset_classification import Generic_WSI_Classification_Dataset
from tadgraph.datasets.dataset_survival_prediction import Generic_MIL_Survival_Dataset
from tadgraph.datasets.utils import save_splits

from tadgraph.paths import *

'''
python create_splits_cv.py --dataset tcga_esca --task survival_prediction 
python create_splits_cv.py --dataset tcga_rcc --task survival_prediction 
python create_splits_cv.py --dataset tcga_brca --task survival_prediction 
python create_splits_cv.py --dataset tcga_nsclc --task tumor_subtyping
'''

parser = argparse.ArgumentParser(
    description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5,
                    help='number of splits (default: 5)')
parser.add_argument('--dataset', type=str, default='tcga_nsclc',
                    choices=['camelyon16', 'tcga_nsclc', 'tcga_rcc', 'tcga_kica', 'tcga_brca',
                             'tcga_blca', 'tcga_ucec', 'tcga_esca', 'tcga_prad'])
parser.add_argument('--task', type=str, default="survival_prediction",
                    choices=["tumor_classification", "tumor_subtyping", "staging", "survival_prediction",
                             "her2", "gleason"])
parser.add_argument("--feat_dir", type=str,
                    default="extracted_mag20x_patch512/kimianet_pt_patch_features/pt_files")
parser.add_argument("--subtype", action='store',
                    type=str, nargs='*', default=[])
parser.add_argument('--val_frac', type=float, default=0.2,
                    help='fraction of labels for validation (default: 0.2)')
parser.add_argument('--test_frac', type=float, default=0.2,
                    help='fraction of labels for test (default: 0.2)')
parser.add_argument("--custom_test", action="store_true")

args = parser.parse_args()


# if args.task == "survival_prediction":
#     args.val_frac = 0.

if args.dataset.lower() == "camelyon16":
    # # we only need 1 fold
    # args.n_classes = 2
    # if args.custom_test:
    #     args.k = 1

    # dataset = Generic_WSI_Classification_Dataset(
    #     csv_path='../dataset_csv/camelyon16/tumor_vs_normal.csv',
    #     shuffle=True,
    #     seed=args.seed,
    #     print_info=True,
    #     label_dict={'normal': 0, 'tumor': 1},
    #     patient_strat=True,
    #     ignore=[],
    #     test_frac=args.test_frac,
    #     custom_test=args.custom_test)
    args.n_classes = 2
    if args.task == "tumor_classification":
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'camelyon16/tumor_vs_normal.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            label_dict={'normal': 0, 'tumor': 1},
            patient_strat=True,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)

elif args.dataset.lower() == "tcga_nsclc":
    args.n_classes = 2
    if args.task == "tumor_subtyping":
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_nsclc/tumor_subtyping.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            label_dict={'LUAD': 0, 'LUSC': 1},
            patient_strat=True,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    elif args.task in ["staging", "survival_prediction"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_nsclc/survival_prediction.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="cancer_stage",
            label_dict={'early_stage': 0, 'late_stage': 1},
            subtype=args.subtype,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    else:
        raise RuntimeError(f"No task named {args.task}!")

elif args.dataset.lower() == "tcga_rcc":
    args.n_classes = 3
    if args.task == "tumor_subtyping":
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_rcc/tumor_subtyping.csv'),
            data_dir=os.path.join(RCC_DATA_DIR, args.feat_dir),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            label_dict={'KIRC': 0, 'KICH': 1, 'KIRP': 2},
            # label_dict={"KIRP": 0, "KICH": 1},
            subtype=args.subtype,
            patient_strat=True,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    elif args.task in ["staging", "survival_prediction"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_rcc/survival_prediction.csv'),
            data_dir=os.path.join(RCC_DATA_DIR, args.feat_dir),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="cancer_stage",
            label_dict={'early_stage': 0, 'late_stage': 1},
            ignore=[],
            patient_voting='maj',
            subtype=args.subtype,
            test_frac=args.test_frac)
    else:
        raise RuntimeError(f"No task named {args.task}!")

elif args.dataset.lower() == "tcga_kica":
    args.n_classes = 3
    if args.task == "tumor_subtyping":
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_kica/tumor_subtyping.csv'),
            data_dir=os.path.join(KICA_DATA_DIR, args.feat_dir),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            label_dict={"KIRP": 0, "KICH": 1},
            subtype=args.subtype,
            patient_strat=True,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    elif args.task in ["staging", "survival_prediction"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_kica/survival_prediction.csv'),
            data_dir=os.path.join(KICA_DATA_DIR, args.feat_dir),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="cancer_stage",
            label_dict={'early_stage': 0, 'late_stage': 1},
            ignore=[],
            patient_voting='maj',
            subtype=args.subtype,
            test_frac=args.test_frac)
    else:
        raise RuntimeError(f"No task named {args.task}!")

elif args.dataset.lower() == "tcga_brca":
    args.n_classes = 2
    if args.task == "tumor_subtyping":
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_brca/tumor_subtyping.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            label_col="label",
            label_dict={'IDC': 0, 'ILC': 1},
            patient_strat=True,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    elif args.task in ["staging", "survival_prediction"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_brca/survival_prediction.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="cancer_stage",
            label_dict={'early_stage': 0, 'late_stage': 1},
            subtype=args.subtype,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    elif args.task == "her2":
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_brca/her2_status.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            label_col="label",
            label_dict={'Negative': 0, 'Positive': 1},
            patient_strat=True,
            ignore=["Equivocal", "Indeterminate"],
            patient_voting='maj',
            subtype=args.subtype,
            test_frac=args.test_frac
        )
    else:
        raise RuntimeError(f"No task named {args.task}!")

elif args.dataset.lower() == "tcga_esca":
    args.n_classes = 2
    if args.task in ["tumor_subtyping"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_esca/tumor_subtyping.csv'),
            data_dir=os.path.join(ESCA_DATA_DIR, args.feat_dir),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="label",
            label_dict={'squamous': 0, 'adeno': 1},
            subtype=args.subtype,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    elif args.task in ["staging", "survival_prediction"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_esca/survival_prediction.csv'),
            data_dir=os.path.join(ESCA_DATA_DIR, args.feat_dir),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="cancer_stage",
            label_dict={'early_stage': 0, 'late_stage': 1},
            subtype=args.subtype,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    else:
        raise NotImplementedError

elif args.dataset.lower() == "tcga_blca":
    args.n_classes = 2
    if args.task in ["staging", "survival_prediction"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_blca/survival_prediction.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="cancer_stage",
            label_dict={'early_stage': 0, 'late_stage': 1},
            subtype=args.subtype,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    else:
        raise RuntimeError(f"No task named {args.task}!")

elif args.dataset.lower() == "tcga_prad":
    args.n_classes = 2
    if args.task in ["gleason", "staging", "survival_prediction"]:
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=os.path.join(
                DATASET_CSV_DIR, 'tcga_prad/survival_prediction.csv'),
            shuffle=True,
            seed=args.seed,
            print_info=True,
            patient_strat=True,
            label_col="gleason_grade",
            label_dict={6: 0, 7: 1, 8: 2, 9: 3, 10: 4},
            subtype=args.subtype,
            ignore=[],
            patient_voting='maj',
            test_frac=args.test_frac)
    else:
        raise RuntimeError(f"No task named {args.task}!")

num_slides_cls = np.array([len(cls_ids)
                           for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)


def create_cv_splits():
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

    for lf in label_fracs:
        if args.subtype:
            subtype_str = "_".join(args.subtype)
            folder_name = f"TCGA-{subtype_str}_{args.task}_{args.k}fold_val{args.val_frac}_test{dataset.test_frac}_{int(lf*100)}_seed{args.seed}"
        else:
            folder_name = f"{args.dataset}_{args.task}_{args.k}fold_val{args.val_frac}_test{dataset.test_frac}_{int(lf*100)}_seed{args.seed}"
        split_dir = os.path.join("../splits", folder_name)
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k=args.k, val_num=val_num,
                              test_num=test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(
                split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(
                split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(
                split_dir, 'splits_{}_descriptor.csv'.format(i)))


if __name__ == '__main__':
    create_cv_splits()
