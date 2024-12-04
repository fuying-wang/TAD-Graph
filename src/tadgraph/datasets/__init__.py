from argparse import ArgumentParser
import os

from tadgraph.datasets.datamodule import MILDataModule, MILGraphDataModule
from tadgraph.datasets.dataset_classification import Generic_MIL_Classification_Dataset
from tadgraph.datasets.dataset_survival_prediction import Generic_MIL_Survival_Dataset
from tadgraph.paths import *


def create_MIL_dataset(dataset, task, feat_dir, seed, patient_strategy, subtype, use_graph,
                       use_sampling=True, training_data_pct=1.):
    if dataset.lower() == "tcga_nsclc":
        # we can test two tasks on this dataset
        if task.lower() == "subtyping":
            mil_dataset = Generic_MIL_Classification_Dataset(
                data_dir=os.path.join(NSCLC_DATA_DIR, feat_dir),
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_nsclc/tumor_subtyping.csv'),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_dict={'LUAD': 0, 'LUSC': 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph
            )
        elif task.lower() == "staging":
            mil_dataset = Generic_MIL_Classification_Dataset(
                data_dir=os.path.join(NSCLC_DATA_DIR, feat_dir),
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_nsclc/survival_prediction.csv'),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="cancer_stage",
                label_dict={'early_stage': 0, 'late_stage': 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph
            )
        elif task.lower() == "survival_prediction":
            mil_dataset = Generic_MIL_Survival_Dataset(
                data_dir=os.path.join(
                    NSCLC_DATA_DIR, feat_dir),
                csv_path=os.path.join(
                    DATASET_CSV_DIR, 'tcga_nsclc/survival_prediction.csv'),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="survival_days",
                patient_strat=False,
                patient_strategy=patient_strategy,
                subtype=subtype,
                use_graph=use_graph
            )

    elif dataset.lower() == "tcga_rcc":
        if task.lower() == "subtyping":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_rcc/tumor_subtyping.csv'),
                data_dir=os.path.join(RCC_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_dict={'KIRC': 0, 'KICH': 1, 'KIRP': 2},
                # label_dict={"KIRP": 0, "KICH": 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph,
                use_sampling=use_sampling,
                training_data_pct=training_data_pct
            )
        elif task.lower() == "staging":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_rcc/survival_prediction.csv'),
                data_dir=os.path.join(RCC_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="cancer_stage",
                label_dict={'early_stage': 0, 'late_stage': 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph,
                use_sampling=use_sampling,
                training_data_pct=training_data_pct
            )
        elif task.lower() == "survival_prediction":
            # FIXME: fix this part
            mil_dataset = Generic_MIL_Survival_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_rcc/survival_prediction.csv'),
                data_dir=os.path.join(RCC_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="survival_days",
                patient_strat=False,
                patient_strategy=patient_strategy,
                subtype=subtype,
                use_graph=use_graph,
                use_sampling=use_sampling,
                training_data_pct=training_data_pct
            )

    elif dataset.lower() == "tcga_brca":
        if task.lower() == "subtyping":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_brca/tumor_subtyping.csv'),
                data_dir=os.path.join(BRCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_dict={'IDC': 0, 'ILC': 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph
            )
        elif task.lower() == "her2":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_brca/her2_status.csv'),
                data_dir=os.path.join(BRCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="label",
                label_dict={'Negative': 0, 'Positive': 1},
                patient_strat=True,
                ignore=["Equivocal", "Indeterminate"],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph
            )
        elif task.lower() == "survival_prediction":
            # FIXME: fix this part
            mil_dataset = Generic_MIL_Survival_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_brca/survival_prediction.csv'),
                data_dir=os.path.join(BRCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="survival_days",
                patient_strat=False,
                patient_strategy=patient_strategy,
                subtype=subtype,
                use_graph=use_graph
            )
    elif dataset.lower() == "tcga_blca":
        if task.lower() == "staging":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_blca/survival_prediction.csv'),
                data_dir=os.path.join(BLCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="cancer_stage",
                label_dict={'early_stage': 0, 'late_stage': 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph
            )
        elif task.lower() == "survival_prediction":
            mil_dataset = Generic_MIL_Survival_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_blca/survival_prediction.csv'),
                data_dir=os.path.join(BLCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="survival_days",
                patient_strat=False,
                patient_strategy=patient_strategy,
                subtype=subtype,
                use_graph=use_graph
            )
        else:
            raise NotImplementedError
        
    elif dataset.lower() == "tcga_prad":
        if task.lower() == "gleason":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_prad/survival_prediction.csv'),
                data_dir=os.path.join(PRAD_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="gleason_grade",
                label_dict={6: 0, 7: 1, 8: 2, 9: 3},
                patient_strat=True,
                ignore=[10],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph
            )
        elif task.lower() == "survival_prediction":
            mil_dataset = Generic_MIL_Survival_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_prad/survival_prediction.csv'),
                data_dir=os.path.join(PRAD_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="survival_days",
                patient_strat=False,
                patient_strategy=patient_strategy,
                subtype=subtype,
                use_graph=use_graph
            )
        else:
            raise NotImplementedError
        
    elif dataset.lower() == "tcga_esca":
        if task.lower() == "subtyping":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_esca/tumor_subtyping.csv'),
                data_dir=os.path.join(ESCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="label",
                label_dict={'squamous': 0, 'adeno': 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph,
                training_data_pct=training_data_pct
            )
        elif task.lower() == "staging":
            mil_dataset = Generic_MIL_Classification_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_esca/survival_prediction.csv'),
                data_dir=os.path.join(ESCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="cancer_stage",
                label_dict={'early_stage': 0, 'late_stage': 1},
                patient_strat=True,
                ignore=[],
                patient_voting='maj',
                subtype=subtype,
                use_graph=use_graph,
                training_data_pct=training_data_pct
            )
        elif task.lower() == "survival_prediction":
            mil_dataset = Generic_MIL_Survival_Dataset(
                csv_path=os.path.join(
                    TAD_GRAPH_DIR, 'dataset_csv/tcga_esca/survival_prediction.csv'),
                data_dir=os.path.join(ESCA_DATA_DIR, feat_dir),
                shuffle=True,
                seed=seed,
                print_info=True,
                label_col="survival_days",
                patient_strat=False,
                patient_strategy=patient_strategy,
                subtype=subtype,
                use_graph=use_graph
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return mil_dataset


def create_MIL_datamodule(args: ArgumentParser):
    dataset = create_MIL_dataset(
        args.dataset, args.task, args.feat_dir, args.seed, args.patient_strategy, args.subtype,
        args.use_graph, args.use_sampling, args.training_data_pct)

    split_csv_path = os.path.join(
        SPLIT_DIR, args.split_file, f"splits_{args.fold}.csv")
    # if args.task in ["subtyping", "staging"]:
    #     split_csv_path = os.path.join(
    #         SPLIT_DIR, args.split_file, f"splits_{args.fold}.csv")
    # elif args.task == "survival_prediction":
    #     split_csv_path = os.path.join(
    #         SURVIVAL_SPLIT_DIR, args.split_file, f"splits_{args.fold}.csv")
    # else:
    #     raise NotImplementedError

    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                     csv_path=split_csv_path)
    datasets = (train_dataset, val_dataset, test_dataset)

    if args.use_graph:
        datamodule = MILGraphDataModule(
            datasets, args.weighted_sample, args.batch_size, args.num_workers)
    else:
        datamodule = MILDataModule(
            datasets, args.weighted_sample, args.batch_size, args.num_workers)

    return datamodule, dataset.n_classes


# def get_num_classes(dataset, task):
#     if dataset.lower() == "camelyon16":
#         return 2
#     elif dataset.lower() == "tcga_nsclc":
#         return 2
#     elif dataset.lower() == "tcga_rcc":
#         return 3
#     elif dataset.lower() == "tcga_brca":
#         return 2


def get_subtyping(dataset):
    if dataset.lower() == "camelyon16":
        return False
    elif dataset.lower() == "tcga_nsclc":
        return True
    elif dataset.lower() == "tcga_rcc":
        return True
    elif dataset.lower() == "tcga_brca":
        return True
