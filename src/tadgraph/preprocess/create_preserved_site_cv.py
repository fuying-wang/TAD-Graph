'''
This script is used to create preserved site cross validation splits for TCGA datasets.
https://github.com/fmhoward/PreservedSiteCV/tree/main 
'''
import argparse
import pandas as pd
import numpy as np
import cvxpy as cp
import cplex
import ipdb
from tadgraph.paths import *
import random

random.seed(1)
np.random.seed(1)

'''
python create_preserved_site_cv.py --dataset tcga_prad --task gleason --folds 5 --label_column gleason_grade
python create_preserved_site_cv.py --dataset tcga_blca --task survival_prediction
python create_preserved_site_cv.py --dataset tcga_esca --task survival_prediction
python create_preserved_site_cv.py --dataset tcga_nsclc --task survival_prediction --subtype LUAD
'''
parser = argparse.ArgumentParser(description='Create stratified crossfolds')
parser.add_argument('--dataset', type=str, default='tcga_brca')
parser.add_argument('--task', type=str, default="her2")
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--label_column', type=str, default="label")
parser.add_argument('--subtype', nargs="+", type=str, default=None)
args = parser.parse_args()


if args.task == "survival_prediction":
    args.label_column = "cancer_stage"


def generate(data, category, values, crossfolds=3, target_column='CV3',
             patient_column='submitter_id', site_column='SITE', timelimit=100, randomseed=0):
    ''' Generates 3 site preserved cross folds with optimal stratification of category
    Input:
        data: dataframe with slides that must be split into crossfolds.
        category: the column in data to stratify by
        values: a list of possible values within category to include for stratification
        crossfolds: number of crossfolds to split data into
        target_column: name for target column to contain the assigned crossfolds for each patient in the output dataframe
        patient_column: column within dataframe indicating unique identifier for patient
        site_column: column within dataframe indicating designated site for a patient
        timelimit: maximum time to spend solving
    Output:
        dataframe with a new column, 'CV3' that contains values 1 - 3, indicating the assigned crossfold
    '''

    submitters = data[patient_column].unique()
    newData = pd.merge(pd.DataFrame(submitters, columns=[patient_column]), data[[
                       patient_column, category, site_column]], on=patient_column, how='left')
    newData.drop_duplicates(inplace=True)
    uniqueSites = data[site_column].unique()
    n = len(uniqueSites)
    listSet = []
    for v in values:
        listOrder = []
        for s in uniqueSites:
            listOrder += [len(newData[(newData[site_column] == s)
                              & (newData[category] == v)].index)]
        listSet += [listOrder]
    gList = []
    for i in range(crossfolds):
        gList += [cp.Variable(n, boolean=True)]
    A = np.ones(n)
    constraints = [sum(gList) == A]
    error = 0
    for v in range(len(values)):
        for i in range(crossfolds):
            error += cp.square(cp.sum(crossfolds *
                               cp.multiply(gList[i], listSet[v])) - sum(listSet[v]))
    prob = cp.Problem(cp.Minimize(error), constraints)
    prob.solve(solver='CPLEX', cplex_params={
               "timelimit": timelimit, "randomseed": randomseed}, verbose=True)
    gSites = []
    for i in range(crossfolds):
        gSites += [[]]
    for i in range(n):
        for j in range(crossfolds):
            if gList[j].value[i] > 0.5:
                gSites[j] += [uniqueSites[i]]
    for i in range(crossfolds):
        str1 = "Crossfold " + str(i+1) + ": "
        j = 0
        for s in listSet:
            str1 = str1 + values[j] + " - " + \
                str(int(np.dot(gList[i].value, s))) + " "
            j = j + 1
        str1 = str1 + " Sites: " + str(gSites[i])
        print(str1)
    bins = pd.DataFrame()
    for i in range(crossfolds):
        data.loc[data[site_column].isin(gSites[i]), target_column] = str(i+1)
    return data


def main():
    if args.dataset == "tcga_brca":
        if args.task == "her2":
            csv_file = os.path.join(
                DATASET_CSV_DIR, "tcga_brca/her2_status.csv")
        elif args.task in ["survival_prediction", "staging"]:
            csv_file = os.path.join(
                DATASET_CSV_DIR, "tcga_brca/survival_prediction.csv")

    elif args.dataset == "tcga_prad":
        if args.task == "gleason":
            csv_file = os.path.join(
                DATASET_CSV_DIR, "tcga_prad/survival_prediction.csv")

    elif args.dataset == "tcga_esca":
        if args.task == "survival_prediction":
            csv_file = os.path.join(
                DATASET_CSV_DIR, "tcga_esca/survival_prediction.csv")

    elif args.dataset == "tcga_rcc":
        if args.task == "survival_prediction":
            csv_file = os.path.join(
                DATASET_CSV_DIR, "tcga_rcc/survival_prediction.csv")

    elif args.dataset == "tcga_blca":
        if args.task == "survival_prediction":
            csv_file = os.path.join(
                DATASET_CSV_DIR, "tcga_blca/survival_prediction.csv")

    elif args.dataset == "tcga_nsclc":
        if args.task == "survival_prediction":
            csv_file = os.path.join(
                DATASET_CSV_DIR, "tcga_nsclc/survival_prediction.csv")

    else:
        raise NotImplementedError

    df = pd.read_csv(csv_file)
    if args.task == "her2":
        df = df.loc[df[args.label_column].isin(["Positive", "Negative"])]

    if args.task == "gleason":
        # remove 10 from the dataset
        df = df.loc[df[args.label_column].isin([6, 7, 8, 9])]
        df.loc[df[args.label_column].isin([6, 7]), args.label_column] = "low"
        df.loc[df[args.label_column].isin([8, 9]), args.label_column] = "high"

    if args.subtype:
        subtype_list = [f"TCGA-{x}" for x in args.subtype]
        df = df.loc[df["project_id"].isin(subtype_list)]

    df["site"] = df["case_id"].apply(lambda x: x.split("-")[1])
    label_values = df[args.label_column].unique().tolist()
    label_values = [str(x) for x in label_values]

    df = generate(df, args.label_column, label_values,
                  crossfolds=5, patient_column='case_id',
                  site_column='site', target_column="CV5")

    if args.subtype:
        str_subtype = "_".join(args.subtype)
        save_dir = os.path.join(
            tadgraph_DIR, "preserved_site_splits", f"TCGA-{str_subtype}_{args.task}_5fold_seed1")
    else:
        save_dir = os.path.join(
            tadgraph_DIR, "preserved_site_splits", f"{args.dataset}_{args.task}_5fold_seed1")
    os.makedirs(save_dir, exist_ok=True)

    for split in range(1, 6):
        train_val_ids = np.setdiff1d(np.arange(1, 6), split).tolist()
        test_df = df.loc[df["CV5"] == str(split)]
        val_id = np.random.choice(train_val_ids)
        train_ids = np.setdiff1d(train_val_ids, val_id).tolist()
        train_ids = [str(x) for x in train_ids]

        train_df = df.loc[df["CV5"].isin(train_ids)]
        val_df = df.loc[df["CV5"] == str(val_id)]

        splits = [train_df["slide_id"].reset_index(drop=True),
                  val_df["slide_id"].reset_index(drop=True),
                  test_df["slide_id"].reset_index(drop=True)]

        split_df = pd.concat(splits, ignore_index=True, axis=1)
        split_df.columns = ["train", "val", "test"]
        split_df.to_csv(os.path.join(save_dir, f"splits_{split-1}.csv"))

        bool_df = pd.concat(splits, ignore_index=True, axis=0)
        index = bool_df.values.tolist()
        one_hot = np.eye(len(splits)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset)
                               for dset in splits], axis=0)
        bool_df = pd.DataFrame(bool_array, index=index,
                               columns=['train', 'val', 'test'])
        bool_df.to_csv(os.path.join(save_dir, f"splits_{split-1}_bool.csv"))

        # train_df["label"].value_counts()
        labels = df[args.label_column].unique()
        desc_df = pd.DataFrame(np.zeros((len(labels), 3)), index=labels, columns=[
                               "train", "val", "test"])
        for i, label in enumerate(labels):
            desc_df.loc[label, "train"] = int(train_df[args.label_column].value_counts()[
                label])
            desc_df.loc[label, "val"] = int(
                val_df[args.label_column].value_counts()[label])
            desc_df.loc[label, "test"] = int(
                test_df[args.label_column].value_counts()[label])
        desc_df = desc_df.astype(int)
        desc_df.to_csv(os.path.join(
            save_dir, f"splits_{split-1}_descriptor.csv"))


if __name__ == '__main__':
    main()
