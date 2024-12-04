import datetime
import os
from argparse import ArgumentParser, Namespace
import warnings
warnings.filterwarnings("ignore")

import ipdb
import yaml
import numpy as np
import pandas as pd
import wandb
import torch
from dateutil import tz
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers.wandb import WandbLogger

from tadgraph.datasets import create_MIL_datamodule
from tadgraph.paths import *
from tadgraph.utils.callbacks import LogResultsCallback
from tadgraph.models.tad_graph_module import TADGraphModule
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision('high')


def get_module(args):
    if args.model_name == "tadgraph":
        return TADGraphModule
    else:
        raise NotImplementedError

'''
CUDA_VISIBLE_DEVICES=6 python main.py --model_name tadgraph --config tad_graph_config.yaml \
--dataset tcga_brca --task her2 --split_file tcga_brca_her2_5fold_val0.2_test0.2_100_seed1  \
--feat_dir extracted_mag20x_patch256/vits_tcga_pancancer_dino_pt_patch_features/slide_graph --embed_size 384 --use_graph \
--lambda_sup 1 --lambda_info 0.5 --lambda_unif 0.5
'''

def get_args():
    parser = ArgumentParser(description="TAD-Graph for WSI analysis.")
    # common arguments
    parser.add_argument("--model_name", type=str, default="tadgraph")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--experiment_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="tcga_nsclc",
                        choices=["tcga_nsclc", "tcga_rcc", "tcga_brca", "tcga_blca",
                                 "tcga_esca", "tcga_prad"])
    parser.add_argument("--task", type=str, default="staging",
                        choices=["subtyping", "staging", "survival_prediction",
                                 "her2", "gleason"],
                        help="classification means normal vs tumor. subtyping means tumor subtyping classification.")
    parser.add_argument("--subtype", action='store',
                        type=str, nargs='*', default=[])
    parser.add_argument(
        "--feat_dir", type=str, default="extracted_mag20x_patch256/vits_tcga_pancancer_dino_pt_patch_features/pt_files")
    parser.add_argument("--embed_size", type=int, default=384)
    parser.add_argument("--weighted_sample", action="store_true")
    parser.add_argument("--split_file", type=str,
                        default="camelyon16_transmil")
    parser.add_argument('--patient_strategy', type=str, default="first",
                        choices=["first", "concat"])
    parser.add_argument("--use_graph", action="store_true")
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'nll_surv'], default='nll_surv',
                        help='loss function for tasks')
    parser.add_argument('--alpha_surv', type=float, default=0.,
                        help="How much to weigh uncensored patients")
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--training_data_pct", type=float, default=1.)
    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--lambda_info", type=float, default=1.0)
    parser.add_argument("--lambda_unif", type=float, default=0.5)

    args = parser.parse_args()

    return args


def cli_main():
    args = get_args()

    if args.config is not None:
        # add arguments from yaml config
        config_path = os.path.join(TAD_GRAPH_DIR, "models", args.config)
        opt = yaml.load(open(config_path), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = Namespace(**opt)

    module = get_module(args)
    num_folds = 5
    total_seeds = [0, 1, 2]
    if args.task in ["staging", "subtyping", "her2", "gleason"]:
        report_val_accs = np.zeros((len(total_seeds), num_folds))
        report_val_aucs = np.zeros((len(total_seeds), num_folds))
        report_val_f1 = np.zeros((len(total_seeds), num_folds))
        report_accs = np.zeros((len(total_seeds), num_folds))
        report_aucs = np.zeros((len(total_seeds), num_folds))
        report_f1 = np.zeros((len(total_seeds), num_folds))
    elif args.task == "survival_prediction":
        report_val_c_indx = np.zeros((len(total_seeds), num_folds))
        report_c_index = np.zeros((len(total_seeds), num_folds))
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    with open(SUMMARY_CSV, "a") as f:
        for idx_seed, seed in enumerate(total_seeds):
            for idx_fold, fold in enumerate(range(num_folds)):

                args.seed = seed
                args.fold = fold

                seed_everything(args.seed)

                # get current time
                now = datetime.datetime.now(tz.tzlocal())
                extension = now.strftime("%Y_%m_%d_%H_%M_%S")
                extension = f"{args.model_name}_{args.dataset}_{args.task}_{args.fold}_{extension}"
                ckpt_dir = os.path.join(RESULTS_DIR, f"tadgraph/ckpts/{extension}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # define callbacks
                if args.task in ["subtyping", "staging", "gleason", "her2"]:
                    callbacks = [
                        LearningRateMonitor(logging_interval="step"),
                        ModelCheckpoint(monitor="valid_auc", dirpath=ckpt_dir,
                                        save_last=True, mode="max", save_top_k=1),
                        EarlyStopping(monitor="valid_auc", min_delta=0.,
                                      patience=20, verbose=False, mode="max"),
                        LogResultsCallback(save_dir=os.path.join(
                            RESULTS_DIR, f"tadgraph/logs/{extension}"), task=args.task)
                    ]
                elif args.task == "survival_prediction":
                    callbacks = [
                        LearningRateMonitor(logging_interval="step"),
                        ModelCheckpoint(monitor="valid_c_index", dirpath=ckpt_dir,
                                        save_last=True, mode="max", save_top_k=1),
                        EarlyStopping(monitor="valid_c_index", min_delta=0.,
                                      patience=20, verbose=False, mode="max"),
                        LogResultsCallback(save_dir=os.path.join(
                            RESULTS_DIR, f"tadgraph/logs/{extension}"), task=args.task)
                    ]

                # define logger
                wandb_logger = WandbLogger(
                    project="MIL", save_dir=RESULTS_DIR, name=extension)

                trainer = Trainer(
                    max_epochs=args.max_epochs,
                    accelerator="gpu",
                    # precision="bf16",
                    deterministic=False,
                    devices=1,
                    accumulate_grad_batches=8,
                    callbacks=callbacks,
                    logger=wandb_logger
                )

                # create datamodule
                datamodule, args.n_classes = create_MIL_datamodule(args)

                # create model
                model = module(**args.__dict__)
                # train the model
                trainer.fit(model, datamodule=datamodule)
                # test the model
                trainer.test(model, datamodule=datamodule, ckpt_path="best")

                log_str = f"{now}\t{args.model_name}\t{args.task}\t{args.dataset}\tfold: {args.fold}\tseed: {args.seed}\t"
                if args.task in ["staging", "subtyping", "her2", "gleason"]:
                    report_val_accs[idx_seed, idx_fold] = model.report_val_acc
                    report_val_aucs[idx_seed, idx_fold] = model.report_val_auc
                    report_val_f1[idx_seed, idx_fold] = model.report_val_f1
                    report_accs[idx_seed, idx_fold] = model.report_acc
                    report_aucs[idx_seed, idx_fold] = model.report_auc
                    report_f1[idx_seed, idx_fold] = model.report_f1
                    log_str += f"VAL_ACC: {model.report_val_acc:.4f}\tVAL_AUC: {model.report_val_auc:.4f}\tVAL_F1: {model.report_val_f1:.4f}\t"
                    log_str += f"ACC: {model.report_acc:.4f}\tAUC: {model.report_auc:.4f}\tF1: {model.report_f1:.4f}\t"
                elif args.task == "survival_prediction":
                    report_val_c_indx[idx_seed, idx_fold] = model.report_val_c_index
                    report_c_index[idx_seed, idx_fold] = model.report_c_index
                    log_str += f"VAL C index: {model.report_val_c_index:.4f}\t"
                    log_str += f"C index: {model.report_c_index:.4f}"
                else:
                    raise NotImplementedError

                f.write(log_str + "\n")

                wandb.finish()

        # summarize results
        seed_columns = [f'seed{x}' for x in total_seeds]
        fold_columns = [f"fold{x}" for x in range(num_folds)]
        if args.task in ["staging", "subtyping", "her2", "gleason"]:
            acc_df = pd.DataFrame(
                report_accs, index=seed_columns, columns=fold_columns)
            print("Accuracy Dataframe: ")
            print(acc_df)
            mean_acc = acc_df.values.mean()
            std_acc = acc_df.mean(axis=1).std()

            val_acc_df = pd.DataFrame(
                report_val_accs, index=seed_columns, columns=fold_columns)
            mean_val_acc = val_acc_df.values.mean()
            std_val_acc = val_acc_df.mean(axis=1).std()

            auc_df = pd.DataFrame(
                report_aucs, index=seed_columns, columns=fold_columns)
            print("AUROC Dataframe: ")
            print(auc_df)
            mean_auc = auc_df.values.mean()
            std_auc = auc_df.mean(axis=1).std()

            val_auc_df = pd.DataFrame(
                report_val_aucs, index=seed_columns, columns=fold_columns)
            mean_val_auc = val_auc_df.values.mean()
            std_val_auc = val_auc_df.mean(axis=1).std()

            f1_df = pd.DataFrame(
                report_f1, index=seed_columns, columns=fold_columns)
            print("F1 Dataframe: ")
            print(f1_df)
            mean_f1 = f1_df.values.mean()
            std_f1 = f1_df.mean(axis=1).std()

            val_f1_df = pd.DataFrame(
                report_val_f1, index=seed_columns, columns=fold_columns)
            mean_val_f1 = val_f1_df.values.mean()
            std_val_f1 = val_f1_df.mean(axis=1).std()

            log_str = f"{args.dataset}\t{args.task}\t{args.feat_dir}\tACC: {mean_acc:.4f}+-{std_acc:.4f}\tAUC: {mean_auc:.4f}+-{std_auc:.4f}" \
                f"\tF1: {mean_f1:.4f}+-{std_f1:.4f}"
            log_str += f"\tVAL_ACC: {mean_val_acc:.4f}+-{std_val_acc:.4f}\tVAL_AUC: {mean_val_auc:.4f}+-{std_val_auc:.4f}" \
                f"\tVAL_F1: {mean_val_f1:.4f}+-{std_val_f1:.4f}"
            print(log_str)
            f.write("**" + log_str + "\n")

        elif args.task == "survival_prediction":
            c_index_df = pd.DataFrame(
                report_c_index, index=seed_columns, columns=fold_columns)
            print("C index Dataframe:")
            print(c_index_df)
            mean_c_index = c_index_df.values.mean()
            std_c_index = c_index_df.mean(axis=1).std()

            val_c_index_df = pd.DataFrame(
                report_val_c_indx, index=seed_columns, columns=fold_columns)
            mean_val_c_index = val_c_index_df.values.mean()
            std_val_c_index = val_c_index_df.mean(axis=1).std()

            log_str = f"{args.dataset}\t{args.task}\t{args.feat_dir}\tC-index: {mean_c_index:.4f}+-{std_c_index:.4f}"
            log_str += f"\tVAL_C-index: {mean_val_c_index:.4f}+-{std_val_c_index:.4f}"
            print(log_str)
            f.write("**" + log_str + "\n")
        else:
            raise NotImplementedError


if __name__ == "__main__":
    cli_main()
