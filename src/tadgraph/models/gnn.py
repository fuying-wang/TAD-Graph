'''
GNN models are implemented by torch_geometric
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from tadgraph.models.conv_layers import GINConv, GATv2Conv
from tadgraph.models.clam_model import Attn_Net_Gated
from tadgraph.models.utils import preprocess
import ipdb


class DenseGNN(torch.nn.Module):
    def __init__(self, emb_dim, num_classes, gconv_type="gat_v2", initial_dim=256,
                 attention_head_num=1, number_of_layers=1, MLP_layernum=1,
                 average_nodes=2000, pooling_type="mean",
                 atten_hidden_size=256):
        super(DenseGNN, self).__init__()

        # FIXME: update number of layers
        self.preprocess = preprocess(
            emb_dim, initial_dim, attention_head_num, MLP_layernum)

        if gconv_type == "gat_v2":
            self.convs_list = [
                GATv2Conv(
                    in_channels=attention_head_num * initial_dim,
                    out_channels=initial_dim,
                    edge_dim=2*initial_dim)
            ]

            for _ in range(number_of_layers - 1):
                self.convs_list.append(GATv2Conv(
                    in_channels=initial_dim,
                    out_channels=initial_dim,
                    edge_dim=2*initial_dim))
        else:
            raise NotImplementedError
        self.convs_list = nn.Sequential(*self.convs_list)
        self.number_of_layers = number_of_layers
        self.pooling_type = pooling_type

        self.graph_feature_dim = initial_dim * attention_head_num

        if self.pooling_type == "attention":
            self.attention_net = Attn_Net_Gated(
                L=self.graph_feature_dim, D=atten_hidden_size, dropout=True, n_classes=1)

        self.pred_layer = nn.Linear(self.graph_feature_dim, num_classes)
        # self.pred_layer = nn.Sequential(
        #     nn.Linear(self.graph_feature_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(32, num_classes)
        # )

    def gnn_forward(self, graph_data, edge_atten=None):
        # preprocess node feature and edge feature
        x, adj_t, preprocess_edge_attr = self.preprocess(
            graph_data.x, graph_data.adj_t, graph_data.edge_attr, graph_data.batch)
        batch = graph_data.batch
        row, col, _ = adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        graph_data.edge_index = edge_index

        # z = self.gnn_pool(x, graph_data.batch)

        x = self.convs_list[0](x=x, edge_index=edge_index, edge_attr=preprocess_edge_attr,
                               edge_atten=edge_atten)
        x = F.relu(x)
        # z = z + self.gnn_pool(x, graph_data.batch)
        z = self.gnn_pool(x, graph_data.batch)

        for i in range(self.number_of_layers - 1):
            x = F.dropout(x, p=0.2)
            x = self.convs_list[i+1](x=x, edge_index=edge_index, edge_attr=preprocess_edge_attr,
                                     edge_atten=edge_atten)
            x = F.relu(x)
            # z = z + self.gnn_pool(x, graph_data.batch)
            z = self.gnn_pool(x, graph_data.batch)

        return x, z

    def gnn_pool(self, x, batch):
        if self.pooling_type == "mean":
            x_pool = global_mean_pool(x, batch)
        elif self.pooling_type == "add":
            x_pool = global_add_pool(x, batch)
        elif self.pooling_type == "attention":
            A, h = self.attention_net(x)
            A = torch.transpose(A, 1, 0)      # KxN
            A = F.softmax(A, dim=1)           # softmax over N
            x_pool = A @ h
        else:
            raise NotImplementedError

        x_pool = F.dropout(x_pool, p=0.2)

        return x_pool

    def forward(self, graph_data, node_atten=None, edge_atten=None):

        if node_atten is not None:
            # add node attention initially
            graph_data.x = graph_data.x * node_atten

        _, x_pool = self.gnn_forward(
            graph_data, edge_atten=edge_atten)

        logits = self.pred_layer(x_pool)
        probs = F.softmax(logits, dim=-1)

        return x_pool, logits, probs


if __name__ == "__main__":
    model = DenseGNN(1024, num_classes=2, gconv_type="gat_v2", number_of_layers=1,
                     pooling_type="attention")
    print(model)

    import os
    from tadgraph.datasets.dataset_survival_prediction import Generic_MIL_Survival_Dataset
    from tadgraph.datasets.dataset_classification import Generic_MIL_Classification_Dataset
    from tadgraph.datasets.datamodule import MILGraphDataModule
    from tadgraph.paths import *

    dataset = Generic_MIL_Classification_Dataset(
        csv_path=os.path.join(
            tadgraph_DIR, 'dataset_csv/tcga_esca/survival_prediction.csv'),
        data_dir=os.path.join(
            ESCA_DATA_DIR, "extracted_mag20x_patch512/kimianet_pt_patch_features/WSI_graph"),
        shuffle=False,
        seed=42,
        print_info=True,
        label_col="cancer_stage",
        label_dict={'early_stage': 0, 'late_stage': 1},
        patient_strat=True,
        ignore=[],
        patient_voting='maj',
        subtype=[],
        use_graph=True
    )

    split_dir = os.path.join(
        SPLIT_DIR, 'tcga_esca_staging_5fold_val0.2_test0.2_100_seed1')
    i = 0
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                     csv_path='{}/splits_{}.csv'.format(split_dir, i))

    datasets = (train_dataset, val_dataset, test_dataset)
    dm = MILGraphDataModule(datasets, batch_size=1)
    for batch in dm.test_dataloader():
        # graph_data, subgraph_data, labels, event_time, censorship, slide_id = batch
        graph_data, labels, slide_id, _ = batch
        break

    # num_edges = graph_data.edge_latent.shape[1]
    num_edges = 55908
    edge_atten = torch.rand(num_edges, 1)

    out = model(graph_data, edge_atten=edge_atten)
    print(out)
