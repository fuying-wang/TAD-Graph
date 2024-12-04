from typing import List

from pytorch_lightning import LightningDataModule
# from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
import ipdb

from tadgraph.datasets.dataset_classification import collate_MIL
from tadgraph.datasets.utils import init_sampler
from tadgraph.paths import *


class MILDataModule(LightningDataModule):
    def __init__(
        self,
        datasets,
        weighted_sample: bool = False,
        batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = True
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.data_train, self.data_val, self.data_test = datasets

    def train_dataloader(self):
        sampler = init_sampler(self.data_train, True,
                               self.hparams.weighted_sample)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            collate_fn=collate_MIL,
            pin_memory=self.hparams.pin_memory
        )

    def val_dataloader(self):
        sampler = init_sampler(self.data_val, False,
                               self.hparams.weighted_sample)
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            collate_fn=collate_MIL,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self):
        sampler = init_sampler(self.data_test, False,
                               self.hparams.weighted_sample)
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            collate_fn=collate_MIL,
            pin_memory=self.hparams.pin_memory
        )


class MILGraphDataModule(LightningDataModule):
    def __init__(
        self,
        datasets,
        weighted_sample: bool = False,
        batch_size: int = 1,
        num_workers: int = 16,
        pin_memory: bool = True
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.data_train, self.data_val, self.data_test = datasets

    def train_dataloader(self):
        sampler = init_sampler(self.data_train, True,
                               self.hparams.weighted_sample)
        return GraphDataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            pin_memory=self.hparams.pin_memory
        )

    def val_dataloader(self):
        sampler = init_sampler(self.data_val, False,
                               self.hparams.weighted_sample)
        return GraphDataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self):
        sampler = init_sampler(self.data_test, False,
                               self.hparams.weighted_sample)
        return GraphDataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
            pin_memory=self.hparams.pin_memory
        )


if __name__ == "__main__":
    from tadgraph.datasets.dataset_classification import Generic_MIL_Classification_Dataset
    from tadgraph.datasets.dataset_survival_prediction import Generic_MIL_Survival_Dataset

    dataset = Generic_MIL_Survival_Dataset(
        csv_path=os.path.join(
            DATASET_CSV_DIR, 'tcga_esca/survival_prediction.csv'),
        data_dir=os.path.join(
            ESCA_DATA_DIR, "H2MIL"),
        shuffle=True,
        seed=1,
        print_info=True,
        patient_strat=True,
        subtype=[],
        use_graph=True
    )
    split_dir = os.path.join(
        SPLIT_DIR, 'tcga_esca_survival_prediction_5fold_val0.0_test0.2_100_seed1')

    i = 0
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                     csv_path='{}/splits_{}.csv'.format(split_dir, i))

    datasets = (train_dataset, val_dataset, test_dataset)
    dm = MILGraphDataModule(datasets, batch_size=1, num_workers=1)
    from tqdm import tqdm
    for batch in tqdm(dm.test_dataloader()):
        print(batch)
        ipdb.set_trace()
