import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tadgraph.models.base_trainer import BaseLightningModule
from torch_geometric.utils import is_undirected
from torch_sparse import transpose

from tadgraph.models.gnn import DenseGNN
from tadgraph.models.utils import ExtractorMLP, reorder_like
from tadgraph.paths import *


class TADGraphModule(BaseLightningModule):
    def __init__(
        self,
        task: str = "subtyping",
        embed_size: int = 768,
        shared_emb_dim: int = 64,
        initial_dim: int = 128,
        attention_head_num: int = 1,
        num_layers: int = 1,
        n_classes: int = 2,
        bag_loss: str = "ce",
        alpha_surv: float = 0.,
        learn_edge_att: bool = False,
        init_r: float = 1.0,
        final_r: float = 0.5,
        decay_interval: int = 5,
        decay_r: float = 0.05,
        gconv: float = "gat_v2",
        pooling_type: str = "attention",
        softmax_temperature: float = 0.2,
        num_trivial_sampling: int = 1,
        lambda_sup: float = 1.,
        lambda_info: float = 1.,
        lambda_unif: float = 0.5,
        lambda_causal: float = 0.0,
        lambda_cont: float = 0.0,
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-4,
        *args,
        **kwargs
    ):

        super().__init__(task=task, n_classes=n_classes, bag_loss=bag_loss, alpha_surv=alpha_surv,
                         learning_rate=learning_rate, weight_decay=weight_decay)
        self.save_hyperparameters()

        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        self.clf = DenseGNN(emb_dim=embed_size,
                            num_classes=n_classes,
                            gconv_type=gconv,
                            initial_dim=initial_dim,
                            attention_head_num=attention_head_num,
                            number_of_layers=num_layers,
                            pooling_type=pooling_type)

        self.causal_extractor = ExtractorMLP(
            self.clf.graph_feature_dim, learn_edge_att=learn_edge_att)

        self.init_r = init_r
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r

    @staticmethod
    def sampling(att_log_logits, training, num_sampling=1):
        temp = 1
        att_berns = []
        if training:
            eps = 1e-10
            num_edges = att_log_logits.shape[0]
            random_noise = torch.zeros(num_edges, num_sampling).type_as(
                att_log_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - \
                torch.log(1.0 - random_noise)
            att_bern = ((att_log_logits + random_noise) / temp).sigmoid()
            for i in range(num_sampling):
                att_berns.append(att_bern[:, i].unsqueeze(1))
        else:
            for _ in range(num_sampling):
                att_bern = (att_log_logits).sigmoid()
                att_berns.append(att_bern)
        return att_berns

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def forward(self, batch, batch_idx, split="train"):
        '''
        Forward of CausalGSAT model
        '''
        graph_data = batch[0]

        # forward step of gnn
        emb, _ = self.clf.gnn_forward(graph_data)

        # compute the logits for each node, (number of nodes, 1)
        att_causal_logits = self.causal_extractor(
            emb, graph_data.edge_index, graph_data.batch)
        causal_atts = self.sampling(
            att_causal_logits, split == "train", num_sampling=1)

        causal_edge_atts = []
        for att in causal_atts:
            if self.hparams.learn_edge_att:
                if is_undirected(graph_data.edge_index):
                    trans_idx, trans_val = transpose(
                        graph_data.edge_index, att, None, None, coalesced=False)
                    trans_val_perm = reorder_like(
                        trans_idx, graph_data.edge_index, trans_val)
                    edge_att = (att + trans_val_perm) / 2
                else:
                    edge_att = att
            else:
                # (number of edges, 1)
                edge_att = self.lift_node_att_to_edge_att(
                    att, graph_data.edge_index)

            causal_edge_atts.append(edge_att.clone())
        causal_edge_atts = torch.cat(causal_edge_atts, dim=0)

        trivial_edge_atts = []
        for att in causal_atts:
            # trivial_att: 1 - att
            att = 1 - att
            if self.hparams.learn_edge_att:
                if is_undirected(graph_data.edge_index):
                    trans_idx, trans_val = transpose(
                        graph_data.edge_index, att, None, None, coalesced=False)
                    trans_val_perm = reorder_like(
                        trans_idx, graph_data.edge_index, trans_val)
                    edge_att = (att + trans_val_perm) / 2
                else:
                    edge_att = att
            else:
                # (number of edges, 1)
                edge_att = self.lift_node_att_to_edge_att(
                    att, graph_data.edge_index)

            trivial_edge_atts.append(edge_att.clone())
        trivial_edge_atts = torch.cat(trivial_edge_atts, dim=1)

        # should also consider edge stochasticity
        causal_feats, causal_logits, _ = self.clf(
            graph_data, node_atten=causal_atts[0], edge_atten=causal_edge_atts)
        # causal_embs = F.normalize(causal_feats, dim=1)

        trivial_feats, trivial_logits, _ = self.clf(
            graph_data, node_atten=1 - causal_atts[0], edge_atten=trivial_edge_atts)
        # trivial_embs = F.normalize(trivial_feats, dim=1)

        # get r
        r = self.get_r(self.decay_interval, self.decay_r, self.current_epoch,
                       init_r=self.init_r, final_r=self.final_r)

        # to minimize the upper bound of mutual information
        info_causal_loss = (causal_atts[0] * torch.log(causal_atts[0] / r + 1e-8) + (1 - causal_atts[0])
                            * torch.log((1 - causal_atts[0]) / (1 - r + 1e-8) + 1e-8)).mean()

        info_loss = info_causal_loss
        info_loss *= self.hparams.lambda_info

        additional_loss_dict = {
            "info_loss": info_loss
        }

        self.log("r", r, prog_bar=True, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=self.hparams.batch_size)

        return additional_loss_dict, causal_logits, trivial_logits

    def shared_step(self, batch, batch_idx, split="train"):
        if self.hparams.task in ["subtyping", "staging", "her2", "gleason"]:
            additional_loss_dict, causal_logits, trivial_logits = self(
                batch, batch_idx, split)
            _, labels, _ = batch

            causal_probs = F.softmax(causal_logits, dim=1)
            supervised_loss = self.loss_fn(causal_logits, labels)

            # TODO: fix this
            labels_uniform = torch.ones([self.hparams.batch_size, self.hparams.n_classes]).type_as(
                causal_logits) / self.hparams.n_classes
            unif_loss = self.kl_loss_fn(F.log_softmax(
                trivial_logits, dim=1), F.softmax(labels_uniform, dim=1))
            unif_loss *= self.hparams.lambda_unif

            total_loss = supervised_loss * self.hparams.lambda_sup + \
                additional_loss_dict["info_loss"] + \
                unif_loss

            loss_dict = {
                "loss": total_loss,
                "supervised_loss": supervised_loss,
                "info_loss": additional_loss_dict["info_loss"],
                "unif_loss": unif_loss
            }

            return loss_dict, causal_probs, labels

        elif self.hparams.task in ["survival_prediction"]:
            additional_loss_dict, causal_logits, trivial_logits = self(
                batch, batch_idx, split)
            _, labels, event_time, censorship, _ = batch

            # causal_probs = F.softmax(causal_logits, dim=1)
            causal_hazards = torch.sigmoid(causal_logits)
            causal_S = torch.cumprod(1 - causal_hazards, dim=1)
            supervised_loss = self.loss_fn(
                hazards=causal_hazards, S=causal_S, Y=labels, c=censorship)

            trivial_hazards = torch.sigmoid(trivial_logits)
            labels_uniform = 0.5 * torch.ones([self.hparams.batch_size, self.hparams.n_classes]).type_as(
                trivial_logits)
            unif_loss = self.kl_loss_fn(
                F.log_softmax(trivial_hazards, dim=1), F.softmax(labels_uniform, dim=1))
            unif_loss *= self.hparams.lambda_unif

            total_loss = supervised_loss + \
                additional_loss_dict["info_loss"] + unif_loss

            loss_dict = {
                "loss": total_loss,
                "supervised_loss": supervised_loss,
                "info_loss": additional_loss_dict["info_loss"],
                "unif_loss": unif_loss,
            }

            # TODO: here make sure batch_size is 1
            risk = -torch.sum(causal_S, dim=1).detach().cpu().numpy()
            event_time = event_time.detach().cpu().numpy()
            censorship = censorship.detach().cpu().numpy()

            return loss_dict, risk, censorship, event_time

        else:
            raise NotImplementedError
