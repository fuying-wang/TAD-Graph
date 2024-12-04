import os
import xml.dom.minidom
from glob import glob
import pandas as pd
from tadgraph.paths import *
import ipdb
import xml.etree.ElementTree as ET
import csv
from tqdm import tqdm


def main():
    data_dir = "/home/r15user2/Documents/MIL_Shapley/ori_WSI/TCGA_BRCA/metadata/supp_clinical"
    df = pd.read_csv(os.path.join(BRCA_DATASET_CSV, "survival_prediction.csv"))
    xml_files = glob(os.path.join(data_dir, "*/*.xml"))

    her2_status = []
    patient_ids = []
    for i, xml_file in tqdm(enumerate(xml_files), total=len(xml_files)):
        # DOMTree = xml.dom.minidom.parse(xml_file)
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            patient_ele = root.findall(
                '{http://tcga.nci/bcr/xml/clinical/brca/2.7}patient')[0]
            element = patient_ele.findall(
                "{http://tcga.nci/bcr/xml/clinical/brca/shared/2.7}lab_proc_her2_neu_immunohistochemistry_receptor_status")[0]
            barcode_element = patient_ele.findall(
                "{http://tcga.nci/bcr/xml/shared/2.7}bcr_patient_barcode")[0]
        except:
            print(f"Can't find her2 in {xml_file}")
            continue

        if element.text is not None:
            her2_status.append(element.text)
            patient_ids.append(barcode_element.text)

    her2_df = pd.DataFrame({
        "patient_ids": patient_ids,
        "her2_status": her2_status
    })

    her2_df.drop_duplicates(subset=["patient_ids"], inplace=True)

    # her2_df["her2_status"].value_counts()

    wsi_dir = "/home/r15user2/Documents/MIL_Shapley/ori_WSI/TCGA_BRCA/WSIs"
    slide_ids, case_ids = [], []
    for wsi_file in os.listdir(wsi_dir):
        slide_id = os.path.splitext(wsi_file)[0]
        case_id = "-".join(slide_id.split("-")[:3])
        slide_ids.append(slide_id)
        case_ids.append(case_id)

    slide_df = pd.DataFrame({
        "case_id": case_ids,
        "slide_id": slide_ids
    })

    merged_df = pd.merge(slide_df, her2_df, left_on="case_id",
                         right_on="patient_ids", how="inner")
    merged_df.drop(columns=["patient_ids"], inplace=True)
    merged_df.rename(columns={"her2_status": "label"}, inplace=True)
    merged_df.to_csv(os.path.join(BRCA_DATASET_CSV,
                     "her2_status.csv"), index=False)


if __name__ == "__main__":
    main()
