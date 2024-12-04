from __future__ import division, print_function

import os
from typing import List
import ipdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from tadgraph.datasets.utils import generate_split, nth


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 n_bins=4,
                 patient_strat=False,
                 label_col=None,
                 eps=1e-6,
                 patient_strategy="first",
                 subtype=[],
                 use_graph=False,
                 use_sampling=False,
                 sampling_patch_num=100,
                 min_patch_num=100,
                 training_data_pct=1.,
                 test_frac=None
                 ):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = data_dir
        self.patient_strategy = patient_strategy
        self.use_graph = use_graph
        self.sampling_patch_num = sampling_patch_num
        self.min_patch_num = min_patch_num
        self.use_sampling = use_sampling
        self.training_data_pct = training_data_pct
        self.test_frac = test_frac

        slide_data = pd.read_csv(csv_path, low_memory=False)
        if self.data_dir:
            slide_data = self.check_slide_data(slide_data)
        assert len(slide_data), "No slides are found!"

        if subtype:
            slide_data = self.select_subtype_data(slide_data, subtype)

        if shuffle:
            slide_data = slide_data.sample(frac=1, random_state=seed)
            slide_data.reset_index(drop=True, inplace=True)

        if not label_col:
            label_col = 'survival_days'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col
        # TODO: check this later
        # if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        # censored: part of data is
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        # quantile-based discretization function
        disc_labels, q_bins = pd.qcut(
            uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(
            patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        # create the mapping: patient -> slide
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})
        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        # slide_data = slide_data.assign(slide_id=slide_data['case_id'])
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                # print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {
            'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + \
            list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]

        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        self.n_classes = len(self.slide_data["disc_label"].value_counts())

    def check_slide_data(self, slide_data: pd.DataFrame):
        # initialize sampling indices for each slide
        self.idx2indices = dict()
        # check if data exists
        existing_indices = []
        for row in slide_data.itertuples():
            slide_id = row.slide_id
            full_path = os.path.join(self.data_dir, f"{slide_id}.pt")
            if os.path.exists(full_path):
                # if not self.use_graph:
                #     features = torch.load(full_path)
                #     # only keep WSI with number of patches > 100
                #     if len(features) >= self.min_patch_num:
                #         existing_indices.append(row.Index)
                #         indices = np.random.choice(
                #             len(features), self.sampling_patch_num, replace=False)
                #         self.idx2indices[slide_id] = indices
                # else:
                existing_indices.append(row.Index)

        slide_data = slide_data.loc[existing_indices]
        slide_data.reset_index(drop=True, inplace=True)

        return slide_data

    @staticmethod
    def select_subtype_data(slide_data: pd.DataFrame, subtype: List):
        if len(subtype) > 0:
            TCGA_subtype = map(lambda x: "TCGA-" + x, subtype)
            slide_data = slide_data.loc[slide_data["project_id"].isin(
                TCGA_subtype), :]
        return slide_data

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(
                self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        # get unique patients
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist(
            )
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {'case_id': patients,
                             'label': np.array(patient_labels)}

    # @staticmethod
    # def df_prep(data, n_bins, ignore, label_col):
    #     mask = data[label_col].isin(ignore)
    #     data = data[~mask]
    #     data.reset_index(drop=True, inplace=True)
    #     disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
    #     return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ")
        print(self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' %
                  (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' %
                  (i, self.slide_cls_ids[i].shape[0]))

    def get_split_from_df(self, all_splits: pd.DataFrame, split_key: str = 'train', scaler=None):
        if split_key in all_splits.columns:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True)
            if len(split) > 0:
                mask = self.slide_data['slide_id'].isin(split.tolist())
                # mask = self.slide_data['case_id'].isin(split.tolist())
                df_slice = self.slide_data[mask].reset_index(drop=True)
                split = Generic_Survival_Split(df_slice, data_dir=self.data_dir, label_col=self.label_col,
                                               patient_dict=self.patient_dict, num_classes=self.num_classes,
                                               patient_strategy=self.patient_strategy, use_graph=self.use_graph,
                                               split=split_key, sampling_patch_num=self.sampling_patch_num, idx2indices=self.idx2indices,
                                               use_sampling=self.use_sampling, training_data_pct=self.training_data_pct)
            else:
                split = None

            return split
        else:
            return None

    def return_splits(self, from_id: bool = True, csv_path: str = None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(
                all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(
                all_splits=all_splits, split_key='val')
            test_split = self.get_split_from_df(
                all_splits=all_splits, split_key='test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'label_frac': label_frac,
            'seed': self.seed,
            'custom_test_ids': self.custom_test_ids
        }

        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids,
                            'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids,
                            'samples': len(self.slide_data)})

        all_train_ids = [[] for _ in range(k)]
        all_val_ids = [[] for _ in range(k)]
        all_test_ids = [[] for _ in range(k)]
        np.random.seed(42)
        for c in range(len(self.patient_cls_ids)):
            num_cls_patients = len(self.patient_cls_ids[c])
            kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)
            cnt = 0
            for train, test in kf.split(self.patient_cls_ids[c]):
                all_test_ids[cnt] += self.patient_cls_ids[c][test].tolist()
                train_val_ids = self.patient_cls_ids[c][train]
                val_ids = np.random.choice(train_val_ids, int(
                    num_cls_patients/k), replace=False)
                train_ids = np.setdiff1d(train_val_ids, val_ids)
                all_val_ids[cnt] += val_ids.tolist()
                all_train_ids[cnt] += train_ids.tolist()
                cnt += 1

        settings.update({
            "all_train_ids": all_train_ids,
            "all_val_ids": all_val_ids,
            "all_test_ids": all_test_ids
        })

        self.split_gen = generate_split(**settings)


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        # TODO: this part might have bug
        features = []
        for slide_id in sorted(slide_ids):
            wsi_path = os.path.join(data_dir, f'{slide_id}.pt')
            # wsi_bag = torch.load(wsi_path, weights_only=True)
            wsi_bag = torch.load(wsi_path)
            features.append(wsi_bag)

        if self.patient_strategy == "first":
            features = features[0]
            slide_ids_str = slide_ids[0]
        else:
            raise NotImplementedError

        # elif self.patient_strategy == "concat":
        #     if isinstance(features[0], torch.Tensor):
        #         # if there are multiple features for each patient
        #         features = torch.cat(features, dim=0)
        #     elif isinstance(features[0], Data):
        #         # if use graph data:
        #         if len(features) == 1:
        #             features = features[0]
        #         else:
        #             # here we merge several WSI graphs into one patient-level graph
        #             new_x_list, new_edge_index_list, new_pos_list = [], [], []
        #             node_num_count = 0
        #             max_x = 0.
        #             for graph_feature in features:
        #                 new_x_list.append(graph_feature.x)
        #                 row, col, _ = graph_feature.adj_t.t().coo()
        #                 edge_index = torch.stack([row, col], dim=0)
        #                 new_edge_index_list.append(edge_index + node_num_count)
        #                 graph_pos = graph_feature.pos.clone()
        #                 graph_pos[:, 0] += max_x
        #                 new_pos_list.append(graph_pos)

        #                 # assume the margin is 100
        #                 max_x += torch.max(graph_feature.pos,
        #                                 dim=0).values[0] + 1e+3
        #                 node_num_count += len(graph_feature.x)

        #             new_x = torch.cat(new_x_list, dim=0)
        #             new_edge_index = torch.cat(new_edge_index_list, dim=1)
        #             new_pos = torch.cat(new_pos_list, dim=0)

        #             patient_graph = Data(
        #                 x=new_x, edge_index=new_edge_index, pos=new_pos)
        #             transform = T.Compose([T.Polar(), T.ToSparseTensor()])
        #             features = transform(patient_graph)
        #     else:
        #         raise NotImplementedError

        #     slide_ids_str = " ".join([x for x in slide_ids])

        if self.data_dir:
            full_path = os.path.join(data_dir, f"{slide_id}.pt")
            # features = torch.load(full_path, weights_only=True)
            features = torch.load(full_path)

            if not self.use_graph:
                if self.use_sampling:
                    # TODO: check if the sampling is correct
                    if self.split == "train":
                        indices = np.random.choice(
                            len(features), self.sampling_patch_num, replace=False)
                    else:
                        indices = self.idx2indices[slide_id]
                    # select features
                    features = features[torch.tensor(indices).long()]

            else:
                if (not hasattr(features, "pos") or features.pos is None) and (
                        not hasattr(features, "centroid") or features.centroid is None):
                    features = features

                else:
                    if not hasattr(features, "pos") or features.pos is None:
                        features.pos = features.centroid
                        pos_transfrom = T.Polar()
                        features = pos_transfrom(features)

                    if not hasattr(features, "adj_t"):
                        transfer = T.ToSparseTensor()
                        features = transfer(features)

            return features, int(label), float(event_time), int(c), slide_ids_str


class Generic_Survival_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, patient_strategy="first", data_dir=None, label_col=None,
                 patient_dict=None, num_classes=2, use_graph=False, split="train", sampling_patch_num=100,
                 idx2indices=dict(), use_sampling=True, training_data_pct=1.):

        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.patient_strategy = patient_strategy
        self.use_graph = use_graph
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        self.sampling_patch_num = sampling_patch_num
        self.idx2indices = idx2indices
        self.split = split
        self.use_sampling = use_sampling
        self.training_data_pct = training_data_pct

        if split == "train":
            # self.slide_data = self.slide_data
            # n = len(self.slide_data)
            slide_cls_ids = []
            for i in range(self.num_classes):
                num_slides = len(self.slide_cls_ids[i])
                sampling_num = int(self.training_data_pct * num_slides)
                slide_cls_ids.append(np.random.choice(
                    self.slide_cls_ids[i], sampling_num, replace=False))

            self.slide_cls_ids = slide_cls_ids

            new_indices = np.hstack(self.slide_cls_ids)
            self.slide_data = self.slide_data.loc[new_indices, :]
            self.slide_data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.slide_data)


if __name__ == "__main__":
    from tadgraph.paths import *
    dataset = Generic_MIL_Survival_Dataset(
        data_dir=os.path.join(
            ESCA_DATA_DIR, "H2MIL"),
        csv_path=os.path.join(
            DATASET_CSV_DIR, 'tcga_esca/survival_prediction.csv'),
        shuffle=True,
        seed=1,
        print_info=True,
        label_col="survival_days",
        patient_strat=True,
        patient_strategy="first",
        subtype=[],
        use_graph=True,
        use_sampling=True
    )

    split_dir = os.path.join(
        SPLIT_DIR, 'tcga_esca_survival_prediction_5fold_val0.0_test0.2_100_seed1')

    i = 0
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                     csv_path='{}/splits_{}.csv'.format(split_dir, i))

    datasets = (train_dataset, val_dataset, test_dataset)

    for batch in train_dataset:
        break
    ipdb.set_trace()
