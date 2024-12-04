import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tadgraph.paths import *
import ipdb


def prepare_camelyon16_csv():
    '''
    create a task csv
    '''
    pass
    # # for training
    # total_types = ["normal", "tumor"]
    # slide_ids, labels = [], []
    # for t in total_types:
    #     cur_dir = os.path.join(CAMELYON16_TRAIN_DIR, t)
    #     for _, filename in enumerate(sorted(os.listdir(cur_dir))):
    #         slide_id = os.path.splitext(filename)[0]
    #         slide_ids.append(slide_id)
    #         labels.append(t)

    # train_df = pd.DataFrame({
    #     "slide_id": slide_ids,
    #     "label": labels
    # })

    # test_df = pd.read_csv(CAMELYON16_REFERENCE_CSV, header=None)
    # test_df = test_df[[0, 1]]
    # test_df.rename({0: "slide_id", 1: "label"}, axis=1, inplace=True)
    # test_df["label"] = test_df["label"].apply(lambda x: x.lower())
    # total_df = pd.concat([train_df, test_df], axis=0)
    # n_slides = len(total_df)

    # case_ids = []
    # for case_id in range(n_slides):
    #     case_ids.append(f"patient_{case_id}")
    # total_df.insert(0, "case_id", case_ids, inplace=True)

    # total_df.to_csv(CAMELYON16_DATASET_CSV, index=False)

    # csv_dir = os.path.dirname(CAMELYON16_DATASET_CSV)
    # os.makedirs(csv_dir, exist_ok=True)

    # test_ref_csv = pd.read_csv(CAMELYON16_REFERENCE_CSV, header=None)
    # wsi_dir = os.path.join(CAMELYON16_DATA_DIR, "WSIs")
    # slide_files = os.listdir(wsi_dir)
    # case_ids, slide_ids, labels = [], [], []
    # for case_id, slide_f in enumerate(slide_files):
    #     filename = os.path.splitext(slide_f)[0]
    #     if filename.startswith("normal"):
    #         labels.append("normal")
    #     elif filename.startswith("tumor"):
    #         labels.append("tumor")
    #     elif filename.startswith("test"):
    #         label = test_ref_csv.loc[test_ref_csv[0] == filename, 1].values[0].lower()
    #         labels.append(label)
    #     slide_ids.append(filename)
    #     case_ids.append(f"patient_{case_id}")

    # df = pd.DataFrame({
    #     "case_id": case_ids,
    #     "slide_id": slide_ids,
    #     "label": labels
    # })
    # df.to_csv(CAMELYON16_DATASET_CSV, index=False)


def prepare_tcga_nsclc_csv():
    csv_dir = os.path.dirname(NSCLC_DATASET_CSV)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    # luad_files = sorted(os.listdir(LUAD_DATA_DIR))
    # lusc_files = sorted(os.listdir(LUSC_DATA_DIR))

    # slide_ids, case_ids, labels = [], [], []
    # for luad_file in luad_files:
    #     files = luad_file.split("-")
    #     case_ids.append(f"patient_{files[2]}")
    #     slide_ids.append(os.path.splitext(luad_file)[0])
    #     labels.append("LUAD")

    # for lusc_file in lusc_files:
    #     files = lusc_file.split("-")
    #     case_ids.append(f"patient_{files[2]}")
    #     slide_ids.append(os.path.splitext(lusc_file)[0])
    #     labels.append("LUSC")

    # df = pd.DataFrame({
    #     "case_id": case_ids,
    #     "slide_id": slide_ids,
    #     "label": labels
    # })
    # df.to_csv(NSCLC_DATASET_CSV, index=False)

    NSCLC_df = pd.read_csv(os.path.join(
        NSCLC_DATASET_CSV, "tumor_subtyping.csv"))

    luad_clinical_tsv = os.path.join(NSCLC_METADATA_DIR, "LUAD/clinical.tsv")
    lusc_clinical_tsv = os.path.join(NSCLC_METADATA_DIR, "LUSC/clinical.tsv")
    luad_clinical_df = pd.read_table(luad_clinical_tsv)
    lusc_clinical_df = pd.read_table(lusc_clinical_tsv)
    df = pd.concat([luad_clinical_df, lusc_clinical_df], axis=0)
    df = df[[
        'case_submitter_id',
        'project_id',
        'days_to_death',
        'days_to_last_follow_up',
        'vital_status',
        'primary_diagnosis',
        'ajcc_pathologic_m',
        'ajcc_pathologic_n',
        'ajcc_pathologic_stage',
        'ajcc_pathologic_t'
    ]]

    df.drop_duplicates(subset=["case_submitter_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    selected_clinical_data = pd.merge(
        NSCLC_df, df, how="left", left_on="case_id", right_on="case_submitter_id")
    selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
    selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)

    primary_diagnosis = []
    stage = []
    survival_days_list, censor_list = [], []
    for row in selected_clinical_data.itertuples():
        # censorship: 
        # 1: alive
        # 0: dead
        if row.vital_status == 'Alive':
            try:
                survival_days = int(row.days_to_last_follow_up)
                censor = 1
            except:
                survival_days = np.nan
                censor = np.nan
        elif row.vital_status == 'Dead':
            try:
                survival_days = int(row.days_to_death)
                censor = 0
            except:
                survival_days = np.nan
                censor = np.nan
        else:
            survival_days = np.nan
            censor = np.nan
        survival_days_list.append(survival_days)
        censor_list.append(censor)
        primary_diagnosis.append(row.primary_diagnosis)
        stage.append(row.ajcc_pathologic_stage)

    selected_clinical_data["survival_days"] = survival_days_list
    selected_clinical_data["censorship"] = censor_list

    selected_clinical_data.dropna(subset=['survival_days'], inplace=True)
    survival_days_list = selected_clinical_data['survival_days'].values

    early_stage = [
        'Stage IIA',
        'Stage IA',
        'Stage I',
        'Stage IIB',
        'Stage IB',
        'Stage II',
    ]

    late_stage = [
        'Stage IIIB',
        'Stage IIIA',
        'Stage III',
        'Stage IIIC',
        'Stage IV',
        'Stage IVA',
    ]

    selected_clinical_data['survival_interval'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    # selected_clinical_data['cancer_classification'] = np.ones(selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data['cancer_stage'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data.reset_index(drop=True, inplace=True)

    for i in range(len(selected_clinical_data)):
        time = selected_clinical_data.loc[i, "survival_days"]

        if time < 730:
            selected_clinical_data.loc[i, "survival_interval"] = 0
        elif time >= 730:
            selected_clinical_data.loc[i, "survival_interval"] = 1
        else:
            selected_clinical_data.loc[i, "survival_interval"] = np.nan

        stage = selected_clinical_data.loc[i, "ajcc_pathologic_stage"]
        if pd.isnull(stage):
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan
            continue

        if stage in early_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "early_stage"
        elif stage in late_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "late_stage"
        else:
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan

    selected_clinical_data.dropna(
        subset=['survival_interval', 'cancer_stage'], inplace=True)
    selected_clinical_data = selected_clinical_data[[
        "case_id", "slide_id", "project_id", "censorship", "survival_days", "survival_interval", "cancer_stage"]]
    selected_clinical_data.reset_index(drop=True, inplace=True)
    print(selected_clinical_data['survival_interval'].value_counts())
    print(selected_clinical_data['cancer_stage'].value_counts())

    ipdb.set_trace()

def prepare_tcga_rcc_csv():
    df = pd.read_csv(os.path.join(RCC_DATASET_CSV, "tumor_subtyping.csv"))
    df["project_id"] = df["label"].apply(lambda x: "TCGA-" + x)
    df.to_csv(os.path.join(RCC_DATASET_CSV, "tumor_subtyping.csv"), index=False)

# def prepare_tcga_rcc_csv():
#     csv_dir = os.path.dirname(RCC_DATASET_CSV)
#     if not os.path.exists(csv_dir):
#         os.makedirs(csv_dir, exist_ok=True)

#     # slide_ids, case_ids, labels = [], [], []

#     # KIRC_manifest_file = "/home/r20user2/data/gdc_download/RCC/gdc_manifest_KIRC.txt"
#     # df = pd.read_csv(KIRC_manifest_file, sep="\t")
#     # for row in df.itertuples():
#     #     slide_file = row.filename
#     #     files = slide_file.split("-")
#     #     case_ids.append("-".join(files[:3]))
#     #     slide_ids.append(os.path.splitext(slide_file)[0])
#     #     labels.append("KIRC")

#     # KICH_manifest_file = "/home/r20user2/data/gdc_download/RCC/gdc_manifest_KICH.txt"
#     # df = pd.read_csv(KICH_manifest_file, sep="\t")
#     # for row in df.itertuples():
#     #     slide_file = row.filename
#     #     files = slide_file.split("-")
#     #     case_ids.append("-".join(files[:3]))
#     #     slide_ids.append(os.path.splitext(slide_file)[0])
#     #     labels.append("KICH")

#     # KIRP_manifest_file = "/home/r20user2/data/gdc_download/RCC/gdc_manifest_KIRP.txt"
#     # df = pd.read_csv(KIRP_manifest_file, sep="\t")
#     # for row in df.itertuples():
#     #     slide_file = row.filename
#     #     files = slide_file.split("-")
#     #     case_ids.append("-".join(files[:3]))
#     #     slide_ids.append(os.path.splitext(slide_file)[0])
#     #     labels.append("KIRP")

#     # df = pd.DataFrame({
#     #     "case_id": case_ids,
#     #     "slide_id": slide_ids,
#     #     "label": labels
#     # })

#     # df.to_csv(os.path.join(RCC_DATASET_CSV, "tumor_subtyping.csv"), index=False)

#     RCC_df = pd.read_csv(os.path.join(
#         RCC_DATASET_CSV, "tumor_subtyping.csv"
#     ))

#     kich_clinical_tsv = os.path.join(RCC_METADATA_DIR, "KICH/clinical.tsv")
#     kirp_clinical_tsv = os.path.join(RCC_METADATA_DIR, "KIRP/clinical.tsv")
#     kirc_clinical_tsv = os.path.join(RCC_METADATA_DIR, "KIRC/clinical.tsv")
#     kich_clinical_df = pd.read_table(kich_clinical_tsv)
#     kirp_clinical_df = pd.read_table(kirp_clinical_tsv)
#     kirc_clinical_df = pd.read_table(kirc_clinical_tsv)
#     df = pd.concat([kich_clinical_df, kirp_clinical_df,
#                    kirc_clinical_df], axis=0)
#     df = df[[
#         'case_submitter_id',
#         'project_id',
#         'days_to_death',
#         'days_to_last_follow_up',
#         'vital_status',
#         'primary_diagnosis',
#         'ajcc_pathologic_m',
#         'ajcc_pathologic_n',
#         'ajcc_pathologic_stage',
#         'ajcc_pathologic_t'
#     ]]
#     df.drop_duplicates(subset=["case_submitter_id"], inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     selected_clinical_data = pd.merge(
#         RCC_df, df, how="left", left_on="case_id", right_on="case_submitter_id")
#     selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
#     selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)

#     primary_diagnosis_set = set()
#     stage_set = set()
#     survival_days_list, censor_list = [], []
#     for row in selected_clinical_data.itertuples():
#         if row.vital_status == 'Alive':
#             try:
#                 survival_days = int(row.days_to_last_follow_up)
#                 censor = 1
#             except:
#                 survival_days = np.nan
#                 censor = np.nan
#         elif row.vital_status == 'Dead':
#             try:
#                 survival_days = int(row.days_to_death)
#                 censor = 0
#             except:
#                 survival_days = np.nan
#                 censor = np.nan
#         else:
#             survival_days = np.nan
#             censor = np.nan
#         survival_days_list.append(survival_days)
#         censor_list.append(censor)
#         primary_diagnosis_set.add(row.primary_diagnosis)
#         stage_set.add(row.ajcc_pathologic_stage)

#     selected_clinical_data["survival_days"] = survival_days_list
#     selected_clinical_data["censorship"] = censor_list

#     selected_clinical_data.dropna(subset=['survival_days'], inplace=True)
#     survival_days_list = selected_clinical_data['survival_days'].values

#     early_stage = [
#         'Stage IIA',
#         'Stage IA',
#         'Stage I',
#         'Stage IIB',
#         'Stage IB',
#         'Stage II',
#     ]

#     late_stage = [
#         'Stage IIIB',
#         'Stage IIIA',
#         'Stage III',
#         'Stage IIIC',
#         'Stage IV',
#         'Stage IVA',
#     ]

#     selected_clinical_data['survival_interval'] = np.ones(
#         selected_clinical_data.shape[0], dtype=int)
#     # selected_clinical_data['cancer_classification'] = np.ones(selected_clinical_data.shape[0], dtype=int)
#     selected_clinical_data['cancer_stage'] = np.ones(
#         selected_clinical_data.shape[0], dtype=int)
#     selected_clinical_data.reset_index(drop=True, inplace=True)

#     for i in range(len(selected_clinical_data)):
#         time = selected_clinical_data.loc[i, "survival_days"]

#         if time < 730:
#             selected_clinical_data.loc[i, "survival_interval"] = 0
#         elif time >= 730:
#             selected_clinical_data.loc[i, "survival_interval"] = 1
#         else:
#             selected_clinical_data.loc[i, "survival_interval"] = np.nan

#         stage = selected_clinical_data.loc[i, "ajcc_pathologic_stage"]
#         if pd.isnull(stage):
#             selected_clinical_data.loc[i, "cancer_stage"] = np.nan
#             continue

#         if stage in early_stage:
#             selected_clinical_data.loc[i, "cancer_stage"] = "early_stage"
#         elif stage in late_stage:
#             selected_clinical_data.loc[i, "cancer_stage"] = "late_stage"
#         else:
#             selected_clinical_data.loc[i, "cancer_stage"] = np.nan

#     selected_clinical_data.dropna(
#         subset=['survival_interval', 'cancer_stage'], inplace=True)
#     selected_clinical_data = selected_clinical_data[[
#         "case_id", "slide_id", "project_id", "censorship", "survival_days", "survival_interval", "cancer_stage"]]
#     selected_clinical_data.reset_index(drop=True, inplace=True)
#     print(selected_clinical_data['survival_interval'].value_counts())
#     print(selected_clinical_data['cancer_stage'].value_counts())

#     selected_clinical_data.to_csv(os.path.join(
#         RCC_DATASET_CSV, "survival_prediction.csv"), index=False)


def prepare_tcga_brca_csv():
    # manifest_file = "/home/r15user2/Documents/MIL_Shapley/data/gdc_download/BRCA/gdc_manifest_BRCA.txt"
    # slide_ids, case_ids, labels = [], [], []
    # df = pd.read_csv(manifest_file, sep="\t")
    # for row in df.itertuples():
    #     slide_file = row.filename
    #     files = slide_file.split("-")
    #     case_ids.append(f"patient_{files[2]}")
    #     slide_ids.append(os.path.splitext(slide_file)[0])

    # metadata = "/home/r20user2/Documents/MIL_Interpretation/data/dataset_csv/tcga_brca_subset.csv.zip"
    # metadf = pd.read_csv(metadata)
    # new_df = metadf[["case_id", "slide_id", "oncotree_code"]]
    # new_df.rename(columns={"oncotree_code": "label"}, inplace=True)
    # new_df["slide_id"] = new_df["slide_id"].apply(lambda x: x[:-4])
    # # new_df.to_csv(BRCA_DATASET_CSV, index=False)
    # # df = pd.read_csv(os.path.join(BRCA_DATASET_CSV, "tumor_subtyping.csv"))
    
    # new_df.to_csv(os.path.join(BRCA_DATASET_CSV, "tumor_subtyping.csv"), index=False)

    brca_clinical_tsv = os.path.join("/home/r15user2/WSI/TCGA_BRCA/metadata", "clinical.tsv")
    brca_clinical_df = pd.read_table(brca_clinical_tsv)
    ipdb.set_trace()
    df = brca_clinical_df[[
        'case_submitter_id',
        'project_id',
        'days_to_death',
        'days_to_last_follow_up',
        'vital_status',
        'primary_diagnosis',
        'ajcc_pathologic_m',
        'ajcc_pathologic_n',
        'ajcc_pathologic_stage',
        'ajcc_pathologic_t'
    ]]
    df.drop_duplicates(subset=["case_submitter_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    case_id, slide_id = [], []
    for s_file in os.listdir(BRCA_WSI_DIR):
        s_id = os.path.splitext(s_file)[0]
        c_id = "-".join(s_id.split("-")[:3])
        slide_id.append(s_id)
        case_id.append(c_id)

    BRCA_df = pd.DataFrame({
        "patient_id": case_id,
        "slide_id": slide_id
    })
    selected_clinical_data = pd.merge(
        BRCA_df, df, how="left", left_on="patient_id", right_on="case_submitter_id")
    selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
    selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)
    selected_clinical_data.rename({"patient_id": "case_id"}, axis=1, inplace=True)

    primary_diagnosis = []
    stage = []
    survival_days_list, censor_list = [], []
    for row in selected_clinical_data.itertuples():
        if row.vital_status == 'Alive':
            try:
                survival_days = int(row.days_to_last_follow_up)
                censor = 1
            except:
                survival_days = np.nan
                censor = np.nan
        elif row.vital_status == 'Dead':
            try:
                survival_days = int(row.days_to_death)
                censor = 0
            except:
                survival_days = np.nan
                censor = np.nan
        else:
            survival_days = np.nan
            censor = np.nan
        survival_days_list.append(survival_days)
        censor_list.append(censor)
        primary_diagnosis.append(row.primary_diagnosis)
        stage.append(row.ajcc_pathologic_stage)

    selected_clinical_data["survival_days"] = survival_days_list
    selected_clinical_data["censorship"] = censor_list

    selected_clinical_data.dropna(subset=['survival_days'], inplace=True)
    survival_days_list = selected_clinical_data['survival_days'].values

    early_stage = [
        'Stage IIA',
        'Stage IA',
        'Stage I',
        'Stage IIB',
        'Stage IB',
        'Stage II',
    ]

    late_stage = [
        'Stage IIIB',
        'Stage IIIA',
        'Stage III',
        'Stage IIIC',
        'Stage IV',
        'Stage IVA',
    ]

    selected_clinical_data['survival_interval'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    # selected_clinical_data['cancer_classification'] = np.ones(selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data['cancer_stage'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data.reset_index(drop=True, inplace=True)

    for i in range(len(selected_clinical_data)):
        time = selected_clinical_data.loc[i, "survival_days"]

        if time < 730:
            selected_clinical_data.loc[i, "survival_interval"] = 0
        elif time >= 730:
            selected_clinical_data.loc[i, "survival_interval"] = 1
        else:
            selected_clinical_data.loc[i, "survival_interval"] = np.nan

        stage = selected_clinical_data.loc[i, "ajcc_pathologic_stage"]
        if pd.isnull(stage):
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan
            continue

        if stage in early_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "early_stage"
        elif stage in late_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "late_stage"
        else:
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan

    selected_clinical_data.dropna(
        subset=['survival_interval', 'cancer_stage'], inplace=True)
    selected_clinical_data = selected_clinical_data[[
        "case_id", "slide_id", "project_id", "primary_diagnosis", "censorship", "survival_days", "survival_interval", "cancer_stage"]]
    selected_clinical_data.reset_index(drop=True, inplace=True)
    print(selected_clinical_data['survival_interval'].value_counts())
    print(selected_clinical_data['cancer_stage'].value_counts())

    print(selected_clinical_data["primary_diagnosis"].value_counts())
    selected_clinical_data["label"] = ""
    selected_clinical_data.loc[selected_clinical_data["primary_diagnosis"] == "Infiltrating duct carcinoma, NOS", "label"] = "IDC"
    selected_clinical_data.loc[selected_clinical_data["primary_diagnosis"] == "Lobular carcinoma, NOS", "label"] = "ILC"

    subtyping_df = selected_clinical_data[["case_id", "slide_id", "label"]]
    subtyping_df = subtyping_df.loc[subtyping_df["label"].isin(["IDC", "ILC"]), :]
    # ipdb.set_trace()
    subtyping_df.to_csv(os.path.join(
        BRCA_DATASET_CSV, "tumor_subtyping.csv"), index=False)
    
    survival_df = selected_clinical_data.loc[selected_clinical_data["label"] == "IDC"]
    survival_df.drop(columns=["label"], inplace=True)
    survival_df.to_csv(os.path.join(
        BRCA_DATASET_CSV, "survival_prediction.csv"), index=False)
    

def prepare_tcga_blca_csv():
    blca_clinical_tsv = os.path.join(BLCA_METADATA_DIR, "clinical.tsv")
    blca_clinical_df = pd.read_table(blca_clinical_tsv)
    df = blca_clinical_df[[
        'case_submitter_id',
        'days_to_death',
        'project_id',
        'days_to_last_follow_up',
        'vital_status',
        'primary_diagnosis',
        'ajcc_pathologic_m',
        'ajcc_pathologic_n',
        'ajcc_pathologic_stage',
        'ajcc_pathologic_t'
    ]]
    df.drop_duplicates(subset=["case_submitter_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    case_id, slide_id = [], []
    for s_file in os.listdir(BLCA_WSI_DIR):
        s_id = os.path.splitext(s_file)[0]
        c_id = "-".join(s_id.split("-")[:3])
        slide_id.append(s_id)
        case_id.append(c_id)

    BLCA_df = pd.DataFrame({
        "patient_id": case_id,
        "slide_id": slide_id
    })

    selected_clinical_data = pd.merge(
        BLCA_df, df, how="left", left_on="patient_id", right_on="case_submitter_id")
    selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
    selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)
    selected_clinical_data.rename({"patient_id": "case_id"}, axis=1, inplace=True)

    primary_diagnosis = []
    stage = []
    survival_days_list, censor_list = [], []
    for row in selected_clinical_data.itertuples():
        if row.vital_status == 'Alive':
            try:
                survival_days = int(row.days_to_last_follow_up)
                censor = 1
            except:
                survival_days = np.nan
                censor = np.nan
        elif row.vital_status == 'Dead':
            try:
                survival_days = int(row.days_to_death)
                censor = 0
            except:
                survival_days = np.nan
                censor = np.nan
        else:
            survival_days = np.nan
            censor = np.nan
        survival_days_list.append(survival_days)
        censor_list.append(censor)
        primary_diagnosis.append(row.primary_diagnosis)
        stage.append(row.ajcc_pathologic_stage)

    selected_clinical_data["survival_days"] = survival_days_list
    selected_clinical_data["censorship"] = censor_list

    selected_clinical_data.dropna(subset=['survival_days'], inplace=True)
    survival_days_list = selected_clinical_data['survival_days'].values

    early_stage = [
        'Stage IIA',
        'Stage IA',
        'Stage I',
        'Stage IIB',
        'Stage IB',
        'Stage II',
    ]

    late_stage = [
        'Stage IIIB',
        'Stage IIIA',
        'Stage III',
        'Stage IIIC',
        'Stage IV',
        'Stage IVA',
    ]

    selected_clinical_data['survival_interval'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    # selected_clinical_data['cancer_classification'] = np.ones(selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data['cancer_stage'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data.reset_index(drop=True, inplace=True)

    for i in range(len(selected_clinical_data)):
        time = selected_clinical_data.loc[i, "survival_days"]

        if time < 730:
            selected_clinical_data.loc[i, "survival_interval"] = 0
        elif time >= 730:
            selected_clinical_data.loc[i, "survival_interval"] = 1
        else:
            selected_clinical_data.loc[i, "survival_interval"] = np.nan

        stage = selected_clinical_data.loc[i, "ajcc_pathologic_stage"]
        if pd.isnull(stage):
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan
            continue

        if stage in early_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "early_stage"
        elif stage in late_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "late_stage"
        else:
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan

    selected_clinical_data.dropna(
        subset=['survival_interval', 'cancer_stage'], inplace=True)
    selected_clinical_data = selected_clinical_data[[
        "case_id", "slide_id", "project_id", "primary_diagnosis", "censorship", "survival_days", "survival_interval", "cancer_stage"]]
    selected_clinical_data.reset_index(drop=True, inplace=True)
    print(selected_clinical_data['survival_interval'].value_counts())
    print(selected_clinical_data['cancer_stage'].value_counts())

    os.makedirs(BLCA_DATASET_CSV, exist_ok=True)
    selected_clinical_data.to_csv(os.path.join(
        BLCA_DATASET_CSV, "survival_prediction.csv"), index=False)

def prepare_tcga_ucec_csv():
    ucec_clinical_tsv = os.path.join(UCEC_METADATA_DIR, "clinical.tsv")
    ucec_clinical_df = pd.read_table(ucec_clinical_tsv)
    df = ucec_clinical_df[[
        'case_submitter_id',
        'days_to_death',
        'days_to_last_follow_up',
        'vital_status',
        'primary_diagnosis',
        'ajcc_pathologic_m',
        'ajcc_pathologic_n',
        'ajcc_pathologic_stage',
        'ajcc_pathologic_t'
    ]]
    df.drop_duplicates(subset=["case_submitter_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    case_id, slide_id = [], []
    for s_file in os.listdir(UCEC_WSI_DIR):
        s_id = os.path.splitext(s_file)[0]
        c_id = "-".join(s_id.split("-")[:3])
        slide_id.append(s_id)
        case_id.append(c_id)

    UCEC_df = pd.DataFrame({
        "patient_id": case_id,
        "slide_id": slide_id
    })

    selected_clinical_data = pd.merge(
        UCEC_df, df, how="left", left_on="patient_id", right_on="case_submitter_id")
    selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
    selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)
    selected_clinical_data.rename({"patient_id": "case_id"}, axis=1, inplace=True)

    primary_diagnosis = []
    stage = []
    survival_days_list, censor_list = [], []
    for row in selected_clinical_data.itertuples():
        if row.vital_status == 'Alive':
            try:
                survival_days = int(row.days_to_last_follow_up)
                censor = 1
            except:
                survival_days = np.nan
                censor = np.nan
        elif row.vital_status == 'Dead':
            try:
                survival_days = int(row.days_to_death)
                censor = 0
            except:
                survival_days = np.nan
                censor = np.nan
        else:
            survival_days = np.nan
            censor = np.nan
        survival_days_list.append(survival_days)
        censor_list.append(censor)
        primary_diagnosis.append(row.primary_diagnosis)
        stage.append(row.ajcc_pathologic_stage)

    selected_clinical_data["survival_days"] = survival_days_list
    selected_clinical_data["censorship"] = censor_list

    selected_clinical_data.dropna(subset=['survival_days'], inplace=True)
    survival_days_list = selected_clinical_data['survival_days'].values

    early_stage = [
        'Stage IIA',
        'Stage IA',
        'Stage I',
        'Stage IIB',
        'Stage IB',
        'Stage II',
    ]

    late_stage = [
        'Stage IIIB',
        'Stage IIIA',
        'Stage III',
        'Stage IIIC',
        'Stage IV',
        'Stage IVA',
    ]

    selected_clinical_data['survival_interval'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    # selected_clinical_data['cancer_classification'] = np.ones(selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data['cancer_stage'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data.reset_index(drop=True, inplace=True)

    for i in range(len(selected_clinical_data)):
        time = selected_clinical_data.loc[i, "survival_days"]

        if time < 730:
            selected_clinical_data.loc[i, "survival_interval"] = 0
        elif time >= 730:
            selected_clinical_data.loc[i, "survival_interval"] = 1
        else:
            selected_clinical_data.loc[i, "survival_interval"] = np.nan

        stage = selected_clinical_data.loc[i, "ajcc_pathologic_stage"]
        if pd.isnull(stage):
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan
            continue

        if stage in early_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "early_stage"
        elif stage in late_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "late_stage"
        else:
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan

    selected_clinical_data.dropna(
        subset=['survival_interval', 'cancer_stage'], inplace=True)
    selected_clinical_data = selected_clinical_data[[
        "case_id", "slide_id", "primary_diagnosis", "censorship", "survival_days", "survival_interval", "cancer_stage"]]
    selected_clinical_data.reset_index(drop=True, inplace=True)
    print(selected_clinical_data['survival_interval'].value_counts())
    print(selected_clinical_data['cancer_stage'].value_counts())

    os.makedirs(UCEC_DATASET_CSV, exist_ok=True)
    selected_clinical_data.to_csv(os.path.join(
        UCEC_DATASET_CSV, "survival_prediction.csv"), index=False)


def prepare_tcga_gbmlgg_csv():
    gbm_clinical_tsv = os.path.join(GBMLGG_METADATA_DIR, "GBM/clinical.tsv")
    lgg_clinical_tsv = os.path.join(GBMLGG_METADATA_DIR, "LGG/clinical.tsv")
    gbm_clinical_df = pd.read_table(gbm_clinical_tsv)
    lgg_clinical_df = pd.read_table(lgg_clinical_tsv)
    df = pd.concat([gbm_clinical_df, lgg_clinical_df], axis=0)

    case_id, slide_id = [], []
    for s_file in os.listdir(GBMLGG_DATA_DIR):
        s_id = os.path.splitext(s_file)[0]
        c_id = "-".join(s_id.split("-")[:3])
        slide_id.append(s_id)
        case_id.append(c_id)

    GBMLGG_df = pd.DataFrame({
        "patient_id": case_id,
        "slide_id": slide_id
    })

    selected_clinical_data = pd.merge(
        GBMLGG_df, df, how="left", left_on="patient_id", right_on="case_submitter_id")
    selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
    selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)
    selected_clinical_data.rename({"patient_id": "case_id"}, axis=1, inplace=True)

    primary_diagnosis = []
    stage = []
    survival_days_list, censor_list = [], []
    for row in selected_clinical_data.itertuples():
        if row.vital_status == 'Alive':
            try:
                survival_days = int(row.days_to_last_follow_up)
                censor = 1
            except:
                survival_days = np.nan
                censor = np.nan
        elif row.vital_status == 'Dead':
            try:
                survival_days = int(row.days_to_death)
                censor = 0
            except:
                survival_days = np.nan
                censor = np.nan
        else:
            survival_days = np.nan
            censor = np.nan
        survival_days_list.append(survival_days)
        censor_list.append(censor)
        primary_diagnosis.append(row.primary_diagnosis)
        stage.append(row.ajcc_pathologic_stage)

    selected_clinical_data["survival_days"] = survival_days_list
    selected_clinical_data["censorship"] = censor_list

    selected_clinical_data.dropna(subset=['survival_days'], inplace=True)
    survival_days_list = selected_clinical_data['survival_days'].values

    early_stage = [
        'Stage IIA',
        'Stage IA',
        'Stage I',
        'Stage IIB',
        'Stage IB',
        'Stage II',
    ]

    late_stage = [
        'Stage IIIB',
        'Stage IIIA',
        'Stage III',
        'Stage IIIC',
        'Stage IV',
        'Stage IVA',
    ]

    selected_clinical_data['survival_interval'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    # selected_clinical_data['cancer_classification'] = np.ones(selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data['cancer_stage'] = np.ones(
        selected_clinical_data.shape[0], dtype=int)
    selected_clinical_data.reset_index(drop=True, inplace=True)

    for i in range(len(selected_clinical_data)):
        time = selected_clinical_data.loc[i, "survival_days"]

        if time < 730:
            selected_clinical_data.loc[i, "survival_interval"] = 0
        elif time >= 730:
            selected_clinical_data.loc[i, "survival_interval"] = 1
        else:
            selected_clinical_data.loc[i, "survival_interval"] = np.nan

        stage = selected_clinical_data.loc[i, "ajcc_pathologic_stage"]
        if pd.isnull(stage):
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan
            continue

        if stage in early_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "early_stage"
        elif stage in late_stage:
            selected_clinical_data.loc[i, "cancer_stage"] = "late_stage"
        else:
            selected_clinical_data.loc[i, "cancer_stage"] = np.nan

    selected_clinical_data.dropna(
        subset=['survival_interval', 'cancer_stage'], inplace=True)
    selected_clinical_data = selected_clinical_data[[
        "case_id", "slide_id", "primary_diagnosis", "censorship", "survival_days", "survival_interval", "cancer_stage"]]
    selected_clinical_data.reset_index(drop=True, inplace=True)
    print(selected_clinical_data['survival_interval'].value_counts())
    print(selected_clinical_data['cancer_stage'].value_counts())

    os.makedirs(GBMLGG_DATASET_CSV, exist_ok=True)
    selected_clinical_data.to_csv(os.path.join(
        GBMLGG_DATASET_CSV, "survival_prediction.csv"), index=False)


def prepare_tcga_esca_csv():
    # esca_clinical_tsv = os.path.join(ESCA_METADATA_DIR, "clinical.tsv")
    # esca_clinical_df = pd.read_table(esca_clinical_tsv)
    # df = esca_clinical_df[[
    #     'case_submitter_id',
    #     'days_to_death',
    #     'days_to_last_follow_up',
    #     'vital_status',
    #     'primary_diagnosis',
    #     'ajcc_pathologic_m',
    #     'ajcc_pathologic_n',
    #     'ajcc_pathologic_stage',
    #     'ajcc_pathologic_t'
    # ]]
    # df.drop_duplicates(subset=["case_submitter_id"], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    
    ESCA_df = pd.read_csv(os.path.join(ESCA_DATASET_CSV, "survival_prediction.csv"))
    # case_id, slide_id = [], []
    # for s_file in os.listdir(ESCA_WSI_DIR):
    #     s_id = os.path.splitext(s_file)[0]
    #     c_id = "-".join(s_id.split("-")[:3])
    #     slide_id.append(s_id)
    #     case_id.append(c_id)

    # ESCA_df = pd.DataFrame({
    #     "patient_id": case_id,
    #     "slide_id": slide_id
    # })

    # selected_clinical_data = pd.merge(
    #     ESCA_df, df, how="left", left_on="case_id", right_on="case_submitter_id")
    # selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
    # selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)
    # selected_clinical_data.rename({"patient_id": "case_id"}, axis=1, inplace=True)

    squamous = [
        "Squamous cell carcinoma, NOS",
        "Basaloid squamous cell carcinoma",
        "Squamous cell carcinoma, keratinizing, NOS"
    ]

    adeno = [
        "Adenocarcinoma, NOS",
        "Tubular adenocarcinoma",
    ]

    ESCA_df.loc[ESCA_df["primary_diagnosis"].isin(squamous), "primary_diagnosis"] = "squamous"
    ESCA_df.loc[ESCA_df["primary_diagnosis"].isin(adeno), "primary_diagnosis"] = "adeno"

    subtyping_df = ESCA_df[["case_id", "slide_id", "primary_diagnosis"]]
    subtyping_df.rename({"primary_diagnosis": "label"}, axis=1, inplace=True)
    subtyping_df["project_id"] = "TCGA-ESCA"
    subtyping_df.to_csv(os.path.join(ESCA_DATASET_CSV, "tumor_subtyping.csv"), index=False)
    # ipdb.set_trace()

    # primary_diagnosis = []
    # stage = []
    # survival_days_list, censor_list = [], []
    # for row in selected_clinical_data.itertuples():
    #     if row.vital_status == 'Alive':
    #         try:
    #             survival_days = int(row.days_to_last_follow_up)
    #             censor = 1
    #         except:
    #             survival_days = np.nan
    #             censor = np.nan
    #     elif row.vital_status == 'Dead':
    #         try:
    #             survival_days = int(row.days_to_death)
    #             censor = 0
    #         except:
    #             survival_days = np.nan
    #             censor = np.nan
    #     else:
    #         survival_days = np.nan
    #         censor = np.nan
    #     survival_days_list.append(survival_days)
    #     censor_list.append(censor)
    #     primary_diagnosis.append(row.primary_diagnosis)
    #     stage.append(row.ajcc_pathologic_stage)

    # selected_clinical_data["survival_days"] = survival_days_list
    # selected_clinical_data["censorship"] = censor_list

    # selected_clinical_data.dropna(subset=['survival_days'], inplace=True)
    # survival_days_list = selected_clinical_data['survival_days'].values

    # early_stage = [
    #     'Stage IIA',
    #     'Stage IA',
    #     'Stage I',
    #     'Stage IIB',
    #     'Stage IB',
    #     'Stage II',
    # ]

    # late_stage = [
    #     'Stage IIIB',
    #     'Stage IIIA',
    #     'Stage III',
    #     'Stage IIIC',
    #     'Stage IV',
    #     'Stage IVA',
    # ]

    # selected_clinical_data['survival_interval'] = np.ones(
    #     selected_clinical_data.shape[0], dtype=int)
    # # selected_clinical_data['cancer_classification'] = np.ones(selected_clinical_data.shape[0], dtype=int)
    # selected_clinical_data['cancer_stage'] = np.ones(
    #     selected_clinical_data.shape[0], dtype=int)
    # selected_clinical_data.reset_index(drop=True, inplace=True)

    # for i in range(len(selected_clinical_data)):
    #     time = selected_clinical_data.loc[i, "survival_days"]

    #     if time < 730:
    #         selected_clinical_data.loc[i, "survival_interval"] = 0
    #     elif time >= 730:
    #         selected_clinical_data.loc[i, "survival_interval"] = 1
    #     else:
    #         selected_clinical_data.loc[i, "survival_interval"] = np.nan

    #     stage = selected_clinical_data.loc[i, "ajcc_pathologic_stage"]
    #     if pd.isnull(stage):
    #         selected_clinical_data.loc[i, "cancer_stage"] = np.nan
    #         continue

    #     if stage in early_stage:
    #         selected_clinical_data.loc[i, "cancer_stage"] = "early_stage"
    #     elif stage in late_stage:
    #         selected_clinical_data.loc[i, "cancer_stage"] = "late_stage"
    #     else:
    #         selected_clinical_data.loc[i, "cancer_stage"] = np.nan

    # selected_clinical_data.dropna(
    #     subset=['survival_interval', 'cancer_stage'], inplace=True)
    # selected_clinical_data = selected_clinical_data[[
    #     "case_id", "slide_id", "primary_diagnosis", "censorship", "survival_days", "survival_interval", "cancer_stage"]]
    # selected_clinical_data.reset_index(drop=True, inplace=True)
    # print(selected_clinical_data['survival_interval'].value_counts())
    # print(selected_clinical_data['cancer_stage'].value_counts())

    # os.makedirs(ESCA_DATASET_CSV, exist_ok=True)
    # selected_clinical_data.to_csv(os.path.join(
    #     ESCA_DATASET_CSV, "survival_prediction.csv"), index=False)


def prepare_tcga_kica_csv():
    ori_df = pd.read_csv("/home/r15user2/Documents/MIL_Shapley/tadgraph/dataset_csv/tcga_kica/KIDNEY_patient_and_label.csv")
    target_dir = os.path.join(KICA_DATA_DIR, "extracted_mag20x_patch512/kimianet_pt_patch_features/pt_files")

    filenames = os.listdir(target_dir)
    case_ids = [] 
    slide_ids = []
    for file in filenames:
        slide_id = os.path.splitext(file)[0]
        case_id = "-".join(slide_id.split("-")[:3])
        slide_ids.append(slide_id)
        case_ids.append(case_id)
    
    df = pd.DataFrame({
        "case_id": case_ids,
        "slide_id": slide_ids
    })
    cur_df = pd.merge(ori_df, df, left_on="patient_id", right_on="case_id", how="inner")
    cur_df.reset_index(drop=True, inplace=True)

    subtyping_df = cur_df[["case_id", "slide_id", "cancer_classification"]]
    typing_label_dict = {
        0: "KIRP",
        1: "KICH"
    }
    subtyping_df["cancer_classification"] = subtyping_df["cancer_classification"].apply(lambda x: typing_label_dict[x])
    subtyping_df.rename(columns={"cancer_classification": "label"}, inplace=True)

    staging_label_dict = {
        0: "early_stage",
        1: "late_stage"
    }
    survival_df = cur_df[["case_id", "slide_id", "censor",
                          "survival_days", "survival_interval", "cancer_stage"]]
    survival_df["cancer_stage"] = survival_df["cancer_stage"].apply(lambda x: staging_label_dict[x])
    survival_df.rename(columns={"censor": "censorship"}, inplace=True)
    # ipdb.set_trace()
    # rcc_typing_df = pd.read_csv(os.path.join(RCC_DATASET_CSV, "tumor_subtyping.csv"))
    # rcc_survival_df = pd.read_csv(os.path.join(RCC_DATASET_CSV, "survival_prediction.csv"))

    # kica_typing_df = pd.merge(ori_df, rcc_typing_df, left_on="patient_id", right_on="case_id", how="left")
    # kica_typing_df.drop(columns=["patient_id"], inplace=True)
    # kica_typing_df.dropna(how="all", inplace=True)
    subtyping_df.to_csv(os.path.join(KICA_DATASET_CSV, "tumor_subtyping.csv"), index=False)

    # kica_survival_df = pd.merge(ori_df["patient_id"], rcc_survival_df, left_on="patient_id", right_on="case_id", how="left")
    # kica_survival_df.dropna(inplace=True)
    # kica_survival_df.drop(columns=["patient_id"], inplace=True)
    # label_dict = {
    #     "early_stage": 0,
    #     "late_stage": 1
    # }
    # kica_survival_df.dropna(subset=["cancer_stage_y"], axis=1, inplace=True)
    # kica_survival_df["cancer_stage_y"] = kica_survival_df["cancer_stage_y"].apply(lambda x: label_dict[x])
    # (kica_survival_df["cancer_stage_x"] == kica_survival_df["cancer_stage_y"])
    # ipdb.set_trace()
    # 
    survival_df.to_csv(os.path.join(KICA_DATASET_CSV, "survival_prediction.csv"), index=False)

def prepare_tcga_prad_csv():
    clinical_df = pd.read_table("/home/r15user2/Documents/MIL_Shapley/ori_WSI/TCGA_PRAD/metadata/clinical.tsv")
    clinical_df.drop_duplicates(subset=["case_id", "case_submitter_id"], inplace=True)

    clinical_df = clinical_df[[
        # "case_id",
        "case_submitter_id",
        'days_to_death',
        'days_to_last_follow_up',
        'vital_status',
        'primary_diagnosis',
        # "gleason_grade_group",
        # "gleason_grade_tertiary",
        # "gleason_patterns_percent",
        # 'ajcc_pathologic_stage',
        "primary_gleason_grade",
        "secondary_gleason_grade"
    ]]
    

    case_id, slide_id = [], []
    for s_file in os.listdir(PRAD_WSI_DIR):
        s_id = os.path.splitext(s_file)[0]
        c_id = "-".join(s_id.split("-")[:3])
        slide_id.append(s_id)
        case_id.append(c_id)

    PRAD_df = pd.DataFrame({
        "patient_id": case_id,
        "slide_id": slide_id
    })
    # PRAD_df["case_len"] = PRAD_df["slide_id"].apply(lambda x: len(x.split(".")))
    # PRAD_df.loc[PRAD_df["case_len"] < 2]
    # clinical_df.loc[clinical_df["case_submitter_id"] == "TCGA-HC-7819"]

    selected_clinical_data = pd.merge(
        PRAD_df, clinical_df, how="left", left_on="patient_id", right_on="case_submitter_id")
    selected_clinical_data.drop(columns=["case_submitter_id"], inplace=True)
    selected_clinical_data = selected_clinical_data.replace('\'--', pd.NA)
    selected_clinical_data.rename({"patient_id": "case_id"}, axis=1, inplace=True)


    survival_days_list, censor_list = [], []
    for row in selected_clinical_data.itertuples():
        if row.vital_status == 'Alive':
            try:
                survival_days = int(row.days_to_last_follow_up)
                censor = 1
            except:
                survival_days = np.nan
                censor = np.nan
        elif row.vital_status == 'Dead':
            try:
                survival_days = int(row.days_to_death)
                censor = 0
            except:
                survival_days = np.nan
                censor = np.nan
        else:
            # for not reported, we assume it is censored data
            try:
                survival_days = int(row.days_to_last_follow_up)
                censor = 1
            except:
                survival_days = np.nan
                censor = np.nan

        survival_days_list.append(survival_days)
        censor_list.append(censor)

    selected_clinical_data["survival_days"] = survival_days_list
    selected_clinical_data["censorship"] = censor_list
    selected_clinical_data[selected_clinical_data["survival_days"].isnull()]
    selected_clinical_data.dropna(subset=['survival_days'], inplace=True)

    selected_clinical_data["primary_gleason_grade"] = selected_clinical_data["primary_gleason_grade"].apply(
        lambda x: int(x[-1])
    )
    selected_clinical_data["secondary_gleason_grade"] = selected_clinical_data["secondary_gleason_grade"].apply(
        lambda x: int(x[-1])
    )
    gleason_grade_list = []
    for row in selected_clinical_data.itertuples():
        gleason_grade_list.append(row.primary_gleason_grade + row.secondary_gleason_grade)
    selected_clinical_data["gleason_grade"] = gleason_grade_list

    selected_clinical_data = selected_clinical_data[[
        "case_id", "slide_id", "censorship", "survival_days", "gleason_grade", 
        "primary_gleason_grade", "secondary_gleason_grade"]]
    selected_clinical_data.reset_index(drop=True, inplace=True)

    print(selected_clinical_data['survival_days'].value_counts())
    print(selected_clinical_data['gleason_grade'].value_counts())

    os.makedirs(PRAD_DATASET_CSV, exist_ok=True)
    selected_clinical_data.to_csv(os.path.join(
        PRAD_DATASET_CSV, "survival_prediction.csv"), index=False)


if __name__ == "__main__":
    prepare_tcga_brca_csv()
    # prepare_tcga_nsclc_csv()
    # prepare_tcga_rcc_csv()
    # prepare_tcga_blca_csv()
    # prepare_tcga_ucec_csv()
    # prepare_tcga_gbmlgg_csv()
    # prepare_tcga_esca_csv()
    # prepare_tcga_kica_csv()
    # prepare_tcga_prad_csv()