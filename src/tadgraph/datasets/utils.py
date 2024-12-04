import collections
import math
from itertools import islice

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (RandomSampler, SequentialSampler,
                              WeightedRandomSampler)


def init_sampler(split_dataset, training: bool = False, weighted_sample: bool = True):
    if training:
        if weighted_sample:
            weights = make_weights_for_balanced_classes_split(split_dataset)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(split_dataset)
    else:
        sampler = SequentialSampler(split_dataset)
    return sampler


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N/len(dataset.slide_cls_ids[c])
                        for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None,
                   all_train_ids=[], all_val_ids=[], all_test_ids=[]):
    # indices = np.arange(samples).astype(int)

    # if custom_test_ids is not None:
    #     indices = np.setdiff1d(indices, custom_test_ids)

    for i in range(n_splits):
        yield all_train_ids[i], all_val_ids[i], all_test_ids[i]

    # for i in range(n_splits):
    #     all_val_ids = []
    #     all_test_ids = []
    #     sampled_train_ids = []

    #     if custom_test_ids is not None:  # pre-built test split, do not need to sample
    #         all_test_ids.extend(custom_test_ids)

    #     for c in range(len(val_num)):
    #         possible_indices = np.intersect1d(
    #             cls_ids[c], indices)     # all indices of this class
    #         val_ids = np.random.choice(
    #             possible_indices, val_num[c], replace=False)  # validation ids

    #         # indices of this class left after validation
    #         remaining_ids = np.setdiff1d(possible_indices, val_ids)
    #         all_val_ids.extend(val_ids)

    #         if custom_test_ids is None:  # sample test split
    #             test_ids = np.random.choice(
    #                 remaining_ids, test_num[c], replace=False)
    #             remaining_ids = np.setdiff1d(remaining_ids, test_ids)
    #             all_test_ids.extend(test_ids)

    #         if label_frac == 1:
    #             sampled_train_ids.extend(remaining_ids)
    #         else:
    #             sample_num = math.ceil(len(remaining_ids) * label_frac)
    #             slice_ids = np.arange(sample_num)
    #             sampled_train_ids.extend(remaining_ids[slice_ids])

    #     yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id']
              for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset)
                               for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index,
                          columns=['train', 'val', 'test'])

    df.to_csv(filename)
    print()
