import os
from typing import Any, Dict, List, Optional, Type

import ipdb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import EarlyStopping


class LogResultsCallback(Callback):
    def __init__(self, save_dir, task):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file = os.path.join(self.save_dir, "exp_results.csv")
        self.task = task

    def shared_batch_end(self, split: str, trainer: Trainer, pl_module: LightningModule,
                         outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        ''' We save the results on the end of each batch'''

        batch_size = pl_module.hparams.batch_size
        with open(self.log_file, "a") as f:
            if self.task in ["classification", "subtyping", "staging", "her2", "gleason"]:
                for bz in range(batch_size):
                    log_str = ""
                    log_str += split
                    log_str += ","
                    slide_id = batch[-1][bz]
                    log_str += slide_id
                    log_str += ","
                    prob = outputs["prob"][bz].detach().cpu().numpy()
                    prob_str = " ".join([str(x) for x in prob])
                    log_str += prob_str
                    log_str += ","
                    label = outputs["labels"][bz].detach().cpu().numpy()
                    label_str = str(label)
                    log_str += label_str
                    log_str += "\n"
                    f.write(log_str)
            elif self.task == "survival_prediction":
                for bz in range(batch_size):
                    log_str = ""
                    log_str += split
                    log_str += ","
                    slide_id = batch[-1][bz]
                    log_str += slide_id
                    log_str += ","
                    risk = outputs["risk"][bz]
                    log_str += str(risk)
                    log_str += ","
                    censorship = outputs["censorship"][bz]
                    log_str += str(censorship)
                    log_str += ","
                    event_time = outputs["event_time"][bz]
                    log_str += str(event_time)
                    log_str += "\n"
                    f.write(log_str)
            else:
                raise NotImplementedError()

    # def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     self.shared_batch_end("train", trainer, pl_module,
    #                           outputs, batch, batch_idx)

    # def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
    #     self.shared_batch_end("valid", trainer, pl_module,
    #                           outputs, batch, batch_idx)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        with open(self.log_file, "w") as f:
            if self.task in ["subtyping", "staging", "classification", "her2", "gleason"]:
                f.write("Split,Slide_id,Probabilities,Labels\n")
            elif self.task in ["survival_prediction"]:
                f.write("Split,Slide_id,Risk_scores,Censorship,Event_time\n")
            else:
                raise NotImplementedError()

    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.shared_batch_end("test", trainer, pl_module,
                              outputs, batch, batch_idx)


class MyEarlyStopping(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= 10:
            super().on_validation_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if trainer.current_epoch >= 10:
            super().on_train_end(trainer, pl_module)
