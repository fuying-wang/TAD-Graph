from typing import Any, Union, List
import numpy as np
from scipy import stats
import cv2
from torchmetrics import AUROC, Accuracy, F1Score

import ipdb
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
# from lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score
from tadgraph.utils.survival_loss import init_survival_loss
from tadgraph.utils.patch_localization_utils import non_maxima_suppression, computeFROC


class BaseLightningModule(LightningModule):
    def __init__(self,
                 task: str = "survival_prediction",
                 n_classes: int = 2,
                 bag_loss: str = "nll_surv",
                 alpha_surv: float = 0.,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5) -> None:

        super().__init__()

        if task.lower() in ["subtyping", "staging", "gleason", "her2"]:
            self.loss_fn = nn.CrossEntropyLoss()
        elif task.lower() == "survival_prediction":
            self.loss_fn = init_survival_loss(
                bag_loss, alpha_surv)

        if task in ["subtyping", "staging", "gleason", "her2"]:
            task = "multiclass"
            # define metrics
            # More details in:
            # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
            self.train_acc = Accuracy(
                task=task, num_classes=n_classes, average="micro")
            self.valid_acc = Accuracy(
                task=task, num_classes=n_classes, average="micro")
            self.test_acc = Accuracy(
                task=task, num_classes=n_classes, average="micro")
            self.train_f1 = F1Score(
                task=task, num_classes=n_classes, average="weighted")
            self.valid_f1 = F1Score(
                task=task, num_classes=n_classes, average="weighted")
            self.test_f1 = F1Score(
                task=task, num_classes=n_classes, average="weighted")
            self.train_auroc = AUROC(task=task, num_classes=n_classes)
            self.valid_auroc = AUROC(task=task, num_classes=n_classes)
            self.test_auroc = AUROC(task=task, num_classes=n_classes)

    def shared_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        if self.hparams.task in ["subtyping", "staging", "gleason", "her2"]:
            loss_dict, Y_prob, labels = self.shared_step(
                batch, batch_idx, "train")

            return_dict = {
                "loss": loss_dict["loss"],
                "prob": Y_prob.detach(),
                "labels": labels
            }

            if loss_dict["loss"] is not None:
                # log accuracy
                y_hat = Y_prob.argmax(dim=1)
                self.train_acc(y_hat, labels)
                self.train_f1(y_hat, labels)
                self.train_auroc(Y_prob, labels)
                acc_dict = {
                    "train_acc": self.train_acc,
                    "train_f1": self.train_f1,
                    "train_auc": self.train_auroc,
                }

                self.log_dict(acc_dict, prog_bar=True, on_step=False, on_epoch=True,
                              sync_dist=True, batch_size=self.hparams.batch_size)

        elif self.hparams.task == "survival_prediction":
            loss_dict, risk, censorship, event_time = self.shared_step(
                batch, batch_idx, "train")

            return_dict = {
                "loss": loss_dict["loss"],
                "risk": risk,
                "censorship": censorship,
                "event_time": event_time
            }

        # log loss
        log_dict = {}
        for k, v in loss_dict.items():
            new_k = "train_" + k
            if v is not None:
                log_dict[new_k] = v
            else:
                return None
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True,
                      sync_dist=True, batch_size=self.hparams.batch_size)


        return return_dict

    def validation_step(self, batch, batch_idx):
        if self.hparams.task in ["subtyping", "staging", "gleason", "her2"]:
            loss_dict, Y_prob, labels = self.shared_step(
                batch, batch_idx, "valid")

            return_dict = {
                "loss": loss_dict["loss"],
                "prob": Y_prob.detach(),
                "labels": labels
            }

            if loss_dict["loss"] is not None:
                # log accuracy
                y_hat = Y_prob.argmax(dim=1)
                self.valid_acc(y_hat, labels)
                self.valid_f1(y_hat, labels)
                self.valid_auroc(Y_prob, labels)
                acc_dict = {
                    "valid_acc": self.valid_acc,
                    "valid_f1": self.valid_f1,
                    "valid_auc": self.valid_auroc,
                }
                self.log_dict(acc_dict, prog_bar=True, on_step=False, on_epoch=True,
                              sync_dist=True, batch_size=self.hparams.batch_size)

        elif self.hparams.task == "survival_prediction":
            loss_dict, risk, censorship, event_time = self.shared_step(
                batch, batch_idx, "valid")

            return_dict = {
                "loss": loss_dict["loss"],
                "risk": risk,
                "censorship": censorship,
                "event_time": event_time
            }

        # log loss
        log_dict = {}
        for k, v in loss_dict.items():
            new_k = "val_" + k
            if v is not None:
                log_dict[new_k] = v
            else:
                return None

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True,
                      sync_dist=True, batch_size=self.hparams.batch_size)
        return return_dict

    def test_step(self, batch, batch_idx):
        if self.hparams.task in ["subtyping", "staging", "gleason", "her2"]:
            loss_dict, Y_prob, labels = self.shared_step(
                batch, batch_idx, "test")

            return_dict = {
                "loss": loss_dict["loss"],
                "prob": Y_prob.detach(),
                "labels": labels
            }

            if loss_dict["loss"] is not None:
                # log accuracy
                y_hat = Y_prob.argmax(dim=1)
                self.test_acc(y_hat, labels)
                self.test_f1(y_hat, labels)
                self.test_auroc(Y_prob, labels)
                acc_dict = {
                    "test_acc": self.test_acc,
                    "test_f1": self.test_f1,
                    "test_auc": self.test_auroc,
                }
                self.log_dict(acc_dict, prog_bar=True, on_step=False, on_epoch=True,
                              sync_dist=True, batch_size=self.hparams.batch_size)

        elif self.hparams.task == "survival_prediction":
            loss_dict, risk, censorship, event_time = self.shared_step(
                batch, batch_idx, "test")

            return_dict = {
                "loss": loss_dict["loss"],
                "risk": risk,
                "censorship": censorship,
                "event_time": event_time
            }

        # log loss
        log_dict = {}
        for k, v in loss_dict.items():
            new_k = "test_" + k
            if v is not None:
                log_dict[new_k] = v
            else:
                return None

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True,
                      sync_dist=True, batch_size=self.hparams.batch_size)

        return return_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # FIXME: tune scheduler
        # return optimizer
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 100], gamma=1.)

        scheduler = {
            "scheduler": lr_scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def log_c_index(self, split: str, outputs):
        all_risk_scores, all_censorships, all_event_times = [], [], []
        for output in outputs:
            if output["risk"] is not None:
                all_risk_scores.append(output["risk"])
                all_censorships.append(output["censorship"])
                all_event_times.append(output["event_time"])

        # TODO:
        # here assume batch size is 1
        all_risk_scores = np.hstack(all_risk_scores)
        all_censorships = np.hstack(all_censorships)
        all_event_times = np.hstack(all_event_times)

        if len(outputs) < 10:
            # FIXME: *** sksurv.exceptions.NoComparablePairException: Data has no comparable pairs, cannot estimate concordance index.
            # it only occurs in sanity check
            c_index = 0.
        else:
            c_index = concordance_index_censored(
                (1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        self.log(f"{split}_c_index", c_index, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.batch_size)

        return c_index


    # def on_train_epoch_end(self) -> None:
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if self.hparams.task in ["subtyping", "gleason", "her2"]:
            return super().training_epoch_end(outputs)
        elif self.hparams.task in ["staging"]:
            pred0_num = 0.
            for output in outputs:
                if output["prob"].argmax(dim=1).item() == 0:
                    pred0_num += 1
            self.log("train_pred0_prec", pred0_num/len(outputs), prog_bar=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.hparams.batch_size)
        elif self.hparams.task == "survival_prediction":
            self.log_c_index("train", outputs)

    def on_validation_epoch_start(self) -> None:
        if self.hparams.task in ["subtyping", "staging", "gleason", "her2"]:
            self.valid_epoch_acc = []
            self.valid_epoch_f1 = []
            self.valid_epoch_auc = []
        elif self.hparams.task == "survival_prediction":
            self.valid_epoch_c_index = []

    # def on_validation_epoch_end(self) -> None:
    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        if self.hparams.task in ["subtyping", "staging", "gleason", "her2"]:
            # not calculate metrics when #. samples is fewer than 10
            if len(outputs) < 10:
                return
            probs, labels = [], []

            for output in outputs:
                if output["prob"] is not None:
                    probs.append(output["prob"])
                    labels.append(output["labels"])

            probs = torch.cat(probs, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()

            pred_labels = np.argmax(probs, axis=1)
            val_acc = accuracy_score(labels, pred_labels)
            val_f1 = f1_score(labels, pred_labels, average="weighted")

            if self.hparams.n_classes == 2:
                auc_score = roc_auc_score(labels, probs[:, 1])
            else:
                auc_score = roc_auc_score(
                    labels, probs, multi_class='ovr')
            
            self.valid_epoch_acc.append(val_acc)
            self.valid_epoch_f1.append(val_f1)
            self.valid_epoch_auc.append(auc_score)

        elif self.hparams.task in ["staging"]:
            pred0_num = 0.
            for output in outputs:
                if output["prob"].argmax(dim=1).item() == 0:
                    pred0_num += 1
            self.log("val_pred0_prec", pred0_num/len(outputs), prog_bar=True, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=self.hparams.batch_size)
        elif self.hparams.task == "survival_prediction":
            # self.report_val_c_index = self.log_c_index("valid", outputs)
            if len(outputs) < 10:
                return
            c_index = self.log_c_index("valid", outputs)
            self.valid_epoch_c_index.append(c_index)

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        if self.hparams.task in ["subtyping", "staging", "gleason", "her2"]:
            probs, labels = [], []

            for output in outputs:
                if output["prob"] is not None:
                    probs.append(output["prob"])
                    labels.append(output["labels"])

            probs = torch.cat(probs, dim=0).cpu().numpy()
            labels = torch.cat(labels, dim=0).cpu().numpy()

            pred_labels = np.argmax(probs, axis=1)
            self.report_acc = accuracy_score(labels, pred_labels)
            self.report_f1 = f1_score(
                labels, pred_labels, average="weighted")

            if self.hparams.n_classes == 2:
                self.report_auc = roc_auc_score(labels, probs[:, 1])
            else:
                self.report_auc = roc_auc_score(
                    labels, probs, multi_class='ovr')
            
            # record the maximum value of validation auc
            # self.report_val_auc = np.max(self.valid_auc)
            max_idx = np.argmax(self.valid_epoch_auc)
            self.report_val_auc = self.valid_epoch_auc[max_idx]
            self.report_val_acc = self.valid_epoch_acc[max_idx]
            self.report_val_f1 = self.valid_epoch_f1[max_idx]

        elif self.hparams.task == "survival_prediction":
            self.report_c_index = self.log_c_index("test", outputs)
            self.report_val_c_index = np.max(self.valid_epoch_c_index)