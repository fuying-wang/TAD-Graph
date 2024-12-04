import os
import numpy as np
import pandas as pd
from collections import Counter
import ipdb

def main():
    source_csv_path = "/home/r15user2/Documents/MIL_Shapley/tadgraph/splits/camelyon16_transmil/fold0.csv"
    csv_path = "/home/r15user2/Documents/MIL_Shapley/tadgraph/splits/camelyon16_transmil/splits_0.csv"
    bool_csv_path = "/home/r15user2/Documents/MIL_Shapley/tadgraph/splits/camelyon16_transmil/splits_0_bool.csv"
    desp_csv_path = "/home/r15user2/Documents/MIL_Shapley/tadgraph/splits/camelyon16_transmil/splits_0_descriptor.csv"
    task_csv_path = "/home/r15user2/Documents/MIL_Shapley/tadgraph/dataset_csv/camelyon16/tumor_vs_normal.csv"

    df = pd.read_csv(csv_path)
    print(df)
    new_df = df.loc[:, ["train", "val", "test"]]
    new_df.to_csv(csv_path, index=False)

    train_slide_ids = new_df["train"][new_df["train"].notnull()].values
    train_num = len(train_slide_ids)

    valid_slide_ids = new_df["val"][new_df["val"].notnull()].values
    valid_num = len(valid_slide_ids)

    test_slide_ids = new_df["test"][new_df["test"].notnull()].values
    test_num = len(test_slide_ids) 
    split_slide_ids = [train_slide_ids, valid_slide_ids, test_slide_ids]
    index = np.hstack(split_slide_ids)

    one_hot = np.eye(3).astype(bool)
    bool_array = np.repeat(one_hot, [len(dset) for dset in split_slide_ids], axis=0)
    df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])
    df.to_csv(bool_csv_path)

    task_df = pd.read_csv(task_csv_path)
    labels = sorted(task_df["label"].unique())
    desp_arr = np.zeros((len(labels), len(split_slide_ids))).astype(np.int16)
    for i, slide_ids in enumerate(split_slide_ids):
        cur_labels = []
        for slide_id in slide_ids:
            cur_labels.append(task_df.loc[task_df["slide_id"] == slide_id, "label"].values[0])

        cur_counter = Counter(cur_labels)
        for j, label in enumerate(labels):
            desp_arr[j, i] = cur_counter[label]
    
    desp_df = pd.DataFrame(desp_arr, index=labels, columns=['train', 'val', 'test'])
    desp_df.to_csv(desp_csv_path)


if __name__ == "__main__":
    main()