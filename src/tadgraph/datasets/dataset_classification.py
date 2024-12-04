# Most of this code is from CLAM
from __future__ import division, print_function

import os
import cv2

from typing import List
import h5py
import ipdb
from math import pi as PI
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# from kmeans_pytorch import kmeans
from fast_pytorch_kmeans import KMeans
import torch
from scipy import stats
from torch.utils.data import Dataset
from tadgraph.datasets.utils import generate_split, nth
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 filter_dict={},
                 ignore=[],
                 patient_strat=False,
                 label_col=None,
                 patient_voting='max',
                 test_frac=None,
                 custom_test=False,
                 subtype=[],
                 use_graph=False,
                 use_hetero=False,
                 mag_lst=[],
                 mag_data_dir_lst=[],
                 use_sampling=False,
                 sampling_patch_num=100,
                 min_patch_num=100,
                 training_data_pct=1.,
                 cluster_num=12
                 ):

        super(Generic_WSI_Classification_Dataset, self).__init__()
        self.label_dict = label_dict
        self.n_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        self.custom_test = custom_test
        self.subtype = subtype
        self.use_graph = use_graph
        self.use_hetero = use_hetero
        self.mag_lst = mag_lst
        self.mag_data_dir_lst = mag_data_dir_lst
        self.sampling_patch_num = sampling_patch_num
        self.min_patch_num = min_patch_num
        self.use_sampling = use_sampling
        self.training_data_pct = training_data_pct
        self.cluster_num = cluster_num

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)

        self.data_dir = data_dir
        if self.data_dir:
            slide_data = self.check_slide_data(slide_data)
        assert len(slide_data), "No slides are found!"

        if subtype:
            slide_data = self.select_subtype_data(slide_data, subtype)

        slide_data = self.df_prep(
            slide_data, self.label_dict, ignore, self.label_col)

        # shuffle data
        if shuffle:
            slide_data = slide_data.sample(frac=1, random_state=seed)
            slide_data.reset_index(drop=True, inplace=True)

        self.slide_data = slide_data
        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

        self.task = csv_path.split("/")[-2]
        self.custom_test_ids = None
        if self.custom_test:
            self.prepare_custom_test_ids()

        if self.custom_test_ids is not None:
            test_num = len(self.custom_test_ids)
            self.test_frac = np.round(test_num / len(self.slide_data), 2)
        else:
            self.test_frac = test_frac

    def check_slide_data(self, slide_data: pd.DataFrame):
        # initialize sampling indices for each slide
        self.idx2indices = dict()

        ext_type = self.data_dir.split("/")[-1]
        if ext_type in ["pt_files", "H2MIL", "slide_graph", "TEA_graph_25", "WSI_graph_v2"]:
            # check if data exists
            existing_indices = []
            for row in slide_data.itertuples():
                slide_id = row.slide_id
                full_path = os.path.join(self.data_dir, f"{slide_id}.pt")
                if os.path.exists(full_path):
                    existing_indices.append(row.Index)

            slide_data = slide_data.loc[existing_indices]
            slide_data.reset_index(drop=True, inplace=True)

            return slide_data
        else:
            # we don't need to sample right now
            return slide_data

    @staticmethod
    def select_subtype_data(slide_data: pd.DataFrame, subtype: List):
        if len(subtype) > 0:
            TCGA_subtype = map(lambda x: "TCGA-" + x, subtype)
            slide_data = slide_data.loc[slide_data["project_id"].isin(
                TCGA_subtype), :]
        return slide_data

    def prepare_custom_test_ids(self):
        # we only have customed test ids for camelyon16
        if self.task.lower() == "camelyon16":
            test_ids = []
            for row in self.slide_data.itertuples():
                if row.slide_id.startswith("test"):
                    if self.patient_strat:
                        test_id = np.where(
                            self.patient_data["case_id"] == row.case_id)[0][0]
                        test_ids.append(test_id)
                    else:
                        test_ids.append(row.Index)
            self.custom_test_ids = np.array(test_ids)

    def cls_ids_prep(self):
        # store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.n_classes)]
        for i in range(self.n_classes):
            self.patient_cls_ids[i] = np.where(
                self.patient_data['label'] == i)[0]

        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.n_classes)]
        for i in range(self.n_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        # get unique patients
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []

        for p in patients:
            label = self.slide_data.loc[self.slide_data['case_id'] == p, "label"].values.astype(
                np.int32)
            assert len(label) > 0

            if patient_voting == 'max':
                label = label.max()  # get patient label (MIL convention)
            elif patient_voting == 'maj':
                label = stats.mode(label, keepdims=True)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)

        self.patient_data = {'case_id': patients,
                             'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])

        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.n_classes))
        print("slide-level counts: ")
        print(self.slide_data['label'].value_counts(sort=False))
        for i in range(self.n_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' %
                  (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' %
                  (i, self.slide_cls_ids[i].shape[0]))

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

            if len(self.patient_cls_ids[c]) >= k:
                for train, test in kf.split(self.patient_cls_ids[c]):
                    all_test_ids[cnt] += self.patient_cls_ids[c][test].tolist()
                    train_val_ids = self.patient_cls_ids[c][train]

                    if val_num[c] > 0:
                        val_ids = np.random.choice(train_val_ids, int(
                            num_cls_patients/k), replace=False)
                        train_ids = np.setdiff1d(train_val_ids, val_ids)
                    else:
                        train_ids = train_val_ids.copy()
                        val_ids = self.patient_cls_ids[c][test].copy()

                    all_val_ids[cnt] += val_ids.tolist()
                    all_train_ids[cnt] += train_ids.tolist()
                    cnt += 1

        settings.update({
            "all_train_ids": all_train_ids,
            "all_val_ids": all_val_ids,
            "all_test_ids": all_test_ids
        })

        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]
            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist(
                    )
                    slide_ids[split].extend(slide_indices)

            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Classification_Split(
                df_slice,
                data_dir=self.data_dir,
                num_classes=self.n_classes,
                use_graph=self.use_graph,
                use_hetero=self.use_hetero,
                dataset_name=self.task,
                mag_lst=self.mag_lst,
                mag_data_dir_lst=self.mag_data_dir_lst,
                split=split_key,
                sampling_patch_num=self.sampling_patch_num,
                idx2indices=self.idx2indices,
                use_sampling=self.use_sampling,
                training_data_pct=self.training_data_pct,
                cluster_num=self.cluster_num
            )
        else:
            split = None

        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_MIL_Classification_Dataset(
                df_slice, data_dir=self.data_dir, num_classes=self.n_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(
                    drop=True)
                train_split = Generic_Classification_Split(
                    train_data, num_classes=self.n_classes)

            else:
                train_split = None

            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(
                    drop=True)
                val_split = Generic_Classification_Split(
                    val_data, num_classes=self.n_classes)

            else:
                val_split = None

            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(
                    drop=True)
                test_split = Generic_Classification_Split(
                    test_data, num_classes=self.n_classes)
            else:
                test_split = None

        else:
            assert csv_path
            # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
            all_splits = pd.read_csv(
                csv_path, dtype=self.slide_data['slide_id'].dtype)
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')

        return train_split, val_split, test_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):
        if return_descriptor:
            index = [list(self.label_dict.keys())[
                list(self.label_dict.values()).index(i)] for i in range(self.n_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index,
                              columns=columns)

        count = len(self.train_ids)
        print('\nnumber of training samples: {}'.format(count))
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(
                unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]

        count = len(self.val_ids)
        print('\nnumber of val samples: {}'.format(count))
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(
                unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]

        count = len(self.test_ids)
        print('\nnumber of test samples: {}'.format(count))
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            print('number of samples in cls {}: {}'.format(
                unique[u], counts[u]))
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        # assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)


def find_closet_upper_layer_node(pos, upper_pos_array):
    dist_2 = np.sum((upper_pos_array - pos) ** 2, axis=1)
    return np.argmin(dist_2)


class Generic_MIL_Classification_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, **kwargs):
        super(Generic_MIL_Classification_Dataset, self).__init__(**kwargs)

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        label = torch.tensor(label).long()
        if type(self.data_dir) == dict:
            source = self.slide_data['source'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if self.data_dir or self.mag_data_dir_lst:
            if self.use_hetero:
                assert self.use_graph is True
                assert self.mag_lst is not None
                assert self.mag_data_dir_lst is not None
                thres = 2560

                str_mag_lst = []
                for mag in self.mag_lst:
                    if isinstance(mag, int):
                        str_mag_lst.append(str(mag))
                    elif isinstance(mag, str):
                        str_mag_lst.append(mag)
                    else:
                        raise NotImplementedError
                self.mag_lst = str_mag_lst

                # initial hetero graph
                Hetero_feat = HeteroData()
                # add graph at each resolution
                for res, dir in zip(self.mag_lst, self.mag_data_dir_lst):
                    path = os.path.join(dir, f"{slide_id}.pt")
                    features = torch.load(path)

                    if not hasattr(features, "pos") or features.pos is None:
                        features.pos = features.centroid
                        pos_transfrom = T.Polar()
                        features = pos_transfrom(features)

                    # if not hasattr(features, "adj_t"):
                    #     transfer = T.ToSparseTensor()
                    #     features = transfer(features)

                    Hetero_feat[res].x = features.x
                    Hetero_feat[res].pos = features.pos
                    Hetero_feat[res, "connects",
                                res].edge_attr = features.edge_attr

                    # row, col, _ = features.adj_t.t().coo()
                    # Hetero_feat[res, "connects", res].edge_index = torch.stack(
                    #     [row, col], dim=0
                    # )
                    Hetero_feat[res, "connects",
                                res].edge_index = features.edge_index

                res_lst = sorted(self.mag_lst, reverse=True)
                for i in range(len(res_lst) - 1):
                    bottom_res = res_lst[i]
                    upper_res = res_lst[i + 1]

                    bottom_pos = Hetero_feat[bottom_res].pos  # B
                    upper_pos = Hetero_feat[upper_res].pos  # U
                    dist_matrix = torch.cdist(
                        bottom_pos, upper_pos, p=2)  # B x U
                    min_values, min_indices = torch.min(dist_matrix, dim=1)
                    # upper_indices = min_indices[min_values <= thres]
                    mask = min_values <= thres
                    Hetero_feat[
                        bottom_res, "belongs", upper_res
                    ].edge_index = torch.stack(
                        [torch.arange(len(bottom_pos))[mask],
                         min_indices[mask]], dim=0
                    )
                    # code adopted from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/polar.html
                    (col, row) = Hetero_feat[
                        bottom_res, "belongs", upper_res
                    ].edge_index
                    b_pos, u_pos = bottom_pos[col], upper_pos[row]
                    cart = b_pos - u_pos
                    rho = torch.norm(cart, p=2, dim=-1).view(-1, 1)
                    theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
                    theta = theta + (theta < 0).type_as(theta) * (2 * PI)
                    # normalize
                    rho = rho / rho.max()
                    theta = theta / (2 * PI)
                    polar = torch.cat([rho, theta], dim=-1)
                    Hetero_feat[bottom_res, "belongs",
                                upper_res].edge_attr = polar

                    return Hetero_feat, label, slide_id
            else:
                if (self.dataset_name == "camelyon16") and (not self.use_graph):
                    # for attention-based methods
                    # we have to use h5 file for camelyon16
                    full_path = os.path.join(data_dir, f"{slide_id}.h5")
                    # features = torch.load(full_path, map_location="cpu")
                    if not os.path.exists(full_path):
                        raise RuntimeError

                    with h5py.File(full_path, "r") as f:
                        coords = np.array(f["coords"])
                        coords = torch.tensor(coords).long()
                        features = np.array(f["features"])
                        features = torch.tensor(features)

                    # FIXME: if the size is too large, transmil OOM
                    max_bag_size = 25000
                    if len(features) > max_bag_size:
                        perm = torch.randperm(features.size(0))
                        new_indices = perm[:max_bag_size]
                        features = features[new_indices]
                        coords = coords[new_indices]
                else:
                    full_path = os.path.join(data_dir, f"{slide_id}.pt")
                    features = torch.load(full_path, map_location="cpu")
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

                    coords = features.pos

                if self.dataset_name == "camelyon16":
                    annotation_file = os.path.join(
                        CAMELYON_ANNOTATION_DIR, f"{slide_id}.xml")
                    w, h = self.shape_dict[slide_id]
                    # FIXME: change this later
                    # assume we use 20x, 256 for camelyon16 dataset
                    # Following camelyon16 github, we evaluate the model on level 5
                    level = 5
                    img_anno = np.zeros(
                        (h // (2 ** level), w // (2 ** level)), np.uint8)
                    if os.path.exists(annotation_file):
                        polygons = get_coordinates(annotation_file)
                        for polygon in polygons:
                            polygon = polygon // (2**level)
                            cv2.fillConvexPoly(img_anno, polygon, 255)
                        # cv2.imwrite(os.path.join("./", slide_id + '.jpg'), img_anno)
                        img_anno = img_anno.reshape(
                            (h // (2**level), w // (2**level)))
                        img_anno = img_anno // 255
                    img_anno = torch.tensor(img_anno)

                    return features, label, slide_id, (coords, img_anno)
                else:
                    return features, label, slide_id


class Generic_Classification_Split(Generic_MIL_Classification_Dataset):
    def __init__(
        self,
        slide_data,
        dataset_name="camelyon16",
        data_dir=None,
        num_classes=2,
        use_graph=False,
        use_hetero=False,
        mag_lst=[],
        mag_data_dir_lst=[],
        sampling_patch_num=100,
        idx2indices=dict(),
        use_sampling=True,
        split="train",
        training_data_pct=1.,
        cluster_num=12
    ):
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.n_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.n_classes)]
        self.use_graph = use_graph
        for i in range(self.n_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        self.use_hetero = use_hetero
        self.mag_lst = mag_lst
        self.mag_data_dir_lst = mag_data_dir_lst
        self.sampling_patch_num = sampling_patch_num
        self.idx2indices = idx2indices
        self.split = split
        self.use_sampling = use_sampling
        self.training_data_pct = training_data_pct
        self.cluster_num = cluster_num
        self.dataset_name = dataset_name

        if self.dataset_name == "camelyon16":
            self.shape_dict = read_shapes(CAMELYON16_IMAGE_SHAPE_TXT)

        if split == "train":
            # self.slide_data = self.slide_data
            # n = len(self.slide_data)
            slide_cls_ids = []
            for i in range(self.n_classes):
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


def collate_MIL(batch):
    n_len = len(batch[0])
    if n_len == 3:
        # subtyping or classification
        img = torch.cat([item[0] for item in batch], dim=0)
        label = torch.LongTensor([item[1] for item in batch])
        slide_id = np.array([item[2] for item in batch])

        return img, label, slide_id

    elif n_len == 4:
        # only happens using camelyon16
        img = torch.cat([item[0] for item in batch], dim=0)
        label = torch.LongTensor([item[1] for item in batch])
        slide_id = np.array([item[2] for item in batch])
        coords = torch.cat([item[3][0] for item in batch], dim=0)
        img_anno = torch.cat([item[3][1] for item in batch], dim=0)

        return img, label, slide_id, coords, img_anno

    elif n_len == 5:
        # survival prediction
        img = torch.cat([item[0] for item in batch], dim=0)
        label = torch.LongTensor([item[1] for item in batch])
        event_time = torch.FloatTensor([item[2] for item in batch])
        censorship = torch.LongTensor([item[3] for item in batch])
        slide_id = np.array([item[4] for item in batch])

        return img, label, event_time, censorship, slide_id


if __name__ == "__main__":
    from tadgraph.paths import *

    dataset = Generic_MIL_Classification_Dataset(
        csv_path=os.path.join(
            tadgraph_DIR, 'dataset_csv/tcga_prad/survival_prediction.csv'),
        data_dir=os.path.join(
            PRAD_DATA_DIR, "extracted_mag20x_patch512/kimianet_pt_patch_features/slide_graph"),
        shuffle=True,
        seed=42,
        print_info=True,
        label_col="gleason_grade",
        label_dict={6: 0, 7: 1, 8: 2, 9: 3},
        patient_strat=True,
        ignore=[10],
        patient_voting='maj',
        subtype=[],
        use_graph=True
    )
    split_dir = os.path.join(
        SPLIT_DIR, 'tcga_prad_gleason_5fold_val0.2_test0.2_100_seed1')
    i = 0
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                     csv_path='{}/splits_{}.csv'.format(split_dir, i))

    datasets = (train_dataset, val_dataset, test_dataset)

    # print(len(train_dataset))
    cnt = 0
    for data in test_dataset:
        print(data)
        break
    ipdb.set_trace()
    # print(x.shape)
    # cnt += 1
    # if cnt >= 5:
    #     break
    # print(test_dataset[0][0])
