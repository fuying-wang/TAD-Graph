from typing import Union

import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch_geometric.transforms as T
from torch_geometric.utils import to_torch_coo_tensor
from torch.nn import (Dropout, LayerNorm, LeakyReLU, Linear,
                      PReLU, ReLU)
from torch_geometric.data import Batch, Data
from torch_geometric.nn import BatchNorm, GraphSizeNorm
from torch_geometric.utils import subgraph
from torch_sparse import set_diag

import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm
from torch_geometric.utils import sort_edge_index
from tadgraph.models.clam_model import Attn_Net_Gated


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def decide_loss_type(loss_type, dim):
    if loss_type == "RELU":
        loss_fun = ReLU()
    if loss_type == "Leaky":
        loss_fun = LeakyReLU(negative_slope=0.2)
    elif loss_type == "PRELU":
        loss_fun = PReLU(init=0.2, num_parameters=dim)
    else:
        loss_fun = PReLU(init=0.2, num_parameters=dim)

    return loss_fun


class PreBasicLinear_module(torch.nn.Module):

    def __init__(self, in_f, out_f, dropout_rate, norm_type):

        super(PreBasicLinear_module, self).__init__()
        self.ff = Linear(in_f, out_f)
        if norm_type == "layer":
            self.norm = LayerNorm(out_f)
            self.gbn = None
        else:
            self.norm = BatchNorm(out_f)
            self.gbn = GraphSizeNorm()

        self.act = LeakyReLU(negative_slope=0.2)
        self.drop = Dropout(dropout_rate)
        self.norm_type = norm_type

    def reset_parameters(self):

        self.ff.apply(weight_init)
        self.norm.reset_parameters()

    def forward(self, input_x, batch):

        out_x = self.ff(input_x)
        out_x_temp = 0
        if self.norm_type == "layer":
            # for different graph, use layer norm separately
            for c, item in enumerate(torch.unique(batch)):
                temp = self.norm(out_x[batch == item])
                if c == 0:
                    out_x_temp = temp
                else:
                    out_x_temp = torch.cat((out_x_temp, temp), 0)

        else:
            temp = self.gbn(self.norm(out_x), batch)
            out_x_temp = temp

        out_x = self.act(out_x_temp)
        out_x = self.drop(out_x)

        return out_x


class preprocess(torch.nn.Module):
    def __init__(self, embed_dim, initial_dim, attention_head_num, MLP_layernum,
                 norm_type="layer", simple_distance="N", dropout_rate=0.25):
        super(preprocess, self).__init__()

        prelayerpreset = [
            128, attention_head_num * initial_dim]
        self.prelayernum = []
        self.prelayernum.append(embed_dim)
        for i in range(0, MLP_layernum - 1):
            self.prelayernum.append(prelayerpreset[i])

        dropout_rate = dropout_rate
        norm_type = norm_type

        self.prelayer_last = nn.Sequential(
            Linear(self.prelayernum[0], attention_head_num * initial_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )

        self.edge_position_embedding = nn.Embedding(11, initial_dim)
        self.edge_angle_embedding = nn.Embedding(11, initial_dim)
        self.simple_distance = simple_distance

    def forward(self, x, edge_index, edge_attr, batch, edge_mask=None):
        if (edge_mask == None):
            input_x = x
        else:
            input_x = torch.mul(x, torch.reshape(
                edge_mask, (edge_mask.shape[0], 1)))

        edge_feature = edge_attr
        row, col, _ = edge_index.coo()

        # TODO: here we want to fix the diagonal
        Non_self_feature = edge_feature[~(row == col), :]
        drop_adj_t = set_diag(edge_index)  # add diagonal
        drop_diag_row, drop_diag_col, _ = drop_adj_t.coo()
        drop_edge_attr_diag = np.zeros(
            (drop_diag_row.shape[0], edge_feature.shape[1]))
        drop_edge_attr_diag[
            ~(drop_diag_row == drop_diag_col).cpu().detach().numpy()] = Non_self_feature.cpu().detach().numpy()
        drop_edge_attr = torch.tensor(drop_edge_attr_diag).float()

        if self.simple_distance == "Y":
            pass
        else:
            # create edge embeddings
            trunc_drop_edge_attr = torch.div(
                drop_edge_attr, 0.1, rounding_mode='trunc')
            trunc_drop_edge_attr = trunc_drop_edge_attr.type_as(
                edge_feature).long()

            drop_edge_attr_distance = self.edge_position_embedding(
                trunc_drop_edge_attr[:, 0])
            drop_edge_attr_angle = self.edge_angle_embedding(
                trunc_drop_edge_attr[:, 1])
            drop_edge_attr = torch.cat(
                (drop_edge_attr_distance, drop_edge_attr_angle), 1)

        preprocessed_data = input_x
        preprocessed_data = self.prelayer_last(preprocessed_data)

        return preprocessed_data, drop_adj_t, drop_edge_attr


class PostBasicLinear_module(torch.nn.Module):

    def __init__(self, in_f, out_f, dropout_rate):

        super(PostBasicLinear_module, self).__init__()
        self.Linear = Linear(in_f, out_f)
        self.LeakyReLU = LeakyReLU(negative_slope=0.2)
        self.Dropout = Dropout(dropout_rate)

    def reset_parameters(self):

        self.Linear.apply(weight_init)

    def forward(self, input_x, batch):

        out_x = self.Linear(input_x)
        out_x = self.LeakyReLU(out_x)
        out_x = self.Dropout(out_x)

        return out_x


class postprocess(torch.nn.Module):

    def __init__(self, graph_feature_dim, graph_layer_num, last_input_dim, post_layer_num, dropout_rate):
        super(postprocess, self).__init__()

        poststartlayer = graph_feature_dim * graph_layer_num + last_input_dim
        postlayerpreset = [int(poststartlayer / 2.0),
                           int(poststartlayer / 4.0),
                           int(poststartlayer / 8.0)]
        self.postlayernum = []
        self.postlayernum.append(poststartlayer)
        for i in range(0, post_layer_num):
            self.postlayernum.append(postlayerpreset[i])

        postlayer_list = [PostBasicLinear_module(in_f, out_f, dropout_rate)
                          for in_f, out_f in zip(self.postlayernum, self.postlayernum[1:])]
        self.postlayer_blocks = nn.ModuleList(postlayer_list)

    def reset_parameters(self):

        for i in range(len(self.postlayer_blocks)):
            self.postlayer_blocks[i].reset_parameters()

    def forward(self, input_x, batch):

        postprocessed_data = input_x
        for i in range(len(self.postlayer_blocks)):
            postprocessed_data = self.postlayer_blocks[i](
                postprocessed_data, batch)

        return postprocessed_data


def split_graph(graph_data: Data,
                case_id: str,
                window_size: Union[float, int] = 0.1,
                stride_size: Union[float, int] = 0.05,
                min_node_thres: int = 1) -> Batch:
    # TODO:
    # here we only consider graph dataloader with batch_size = 1
    # another way is to consider a fixed window size and stride size

    row, col, _ = graph_data.adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = graph_data.edge_attr
    graph_pos = graph_data.pos
    max_x, max_y = np.max(graph_pos.cpu().numpy(), axis=0)
    min_x, min_y = np.min(graph_pos.cpu().numpy(), axis=0)
    width = max_x - min_x
    height = max_y - min_y

    if isinstance(window_size, float):
        window_size_xy = (width * window_size, height * window_size)
    elif isinstance(window_size, int):
        window_size_xy = (window_size, window_size)

    if isinstance(stride_size, float):
        stride_size_xy = (width * stride_size, height * stride_size)
    elif isinstance(stride_size, int):
        stride_size_xy = (stride_size, stride_size)

    num_x_grid = int((width - window_size_xy[0]) / stride_size_xy[0]) + 1
    num_y_grid = int((height - window_size_xy[1]) / stride_size_xy[1]) + 1

    subgraph_list = []

    # here we create 19 * 19 subgraphs for each WSI
    for i in range(num_x_grid):
        for j in range(num_y_grid):
            left = i * stride_size_xy[0] + min_x
            right = min(i * stride_size_xy[0] +
                        window_size_xy[0], width) + min_x
            up = j * stride_size_xy[1] + min_y
            bottom = min(j * stride_size_xy[1] +
                         window_size_xy[1], height) + min_y

            node_mask = (graph_pos[:, 0] >= left) & (graph_pos[:, 0] < right) & (
                graph_pos[:, 1] >= up) & (graph_pos[:, 1] < bottom)

            # if the number of subgraphs is smaller than xxx
            if node_mask.cpu().numpy().sum() < min_node_thres:
                continue

            subgraph_edge_index, subgraph_edge_attr = subgraph(
                node_mask, edge_index, edge_attr, relabel_nodes=True)
            subgraph_x = graph_data.x[node_mask]
            subgraph_pos = graph_pos[node_mask]
            subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index,
                                 edge_attr=subgraph_edge_attr, pos=subgraph_pos)
            subgraph_data = T.ToSparseTensor()(subgraph_data)
            subgraph_list.append(subgraph_data)

    subgraph_batch = Batch.from_data_list(subgraph_list)
    return subgraph_batch

def graph_pooling(pooling_type: str, x: torch.Tensor, device = torch.device("cuda:0")):
    ''' pooling_type: mean, sum, attention'''

    if pooling_type == "mean":
        x_pool = x.mean(dim=0, keepdim=True)
    elif pooling_type == "sum":
        x_pool = x.sum(dim=0, keepdim=True)
    elif pooling_type == "attention":
        _, d = x.shape
        attention_net = Attn_Net_Gated(
            L=d, D=256, dropout=True, n_classes=1).to(device)
        A, h = attention_net(x)
        A = torch.transpose(A, 1, 0)      # KxN
        A = F.softmax(A, dim=1)           # softmax over N
        x_pool = A @ h
    else:
        raise RuntimeError(f"No error called {pooling_type}")

    return x_pool


def add_edge_index(graph_data):
    row, col, _ = graph_data.adj_t.t().coo()
    graph_data.edge_index = torch.stack([row, col], dim=0)


def reorder_like(from_edge_index, to_edge_index, values):
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * \
        (to_edge_index.max()+1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError(
            "Edges in from_edge_index and to_edge_index are different, impossible to match both.")
    return values[ranking]


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, learn_edge_att):
        super().__init__()
        self.learn_edge_att = learn_edge_att

        if self.learn_edge_att:
            self.feature_extractor = MLP(
                [hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=0.2)
        else:
            self.feature_extractor = MLP(
                [hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.2)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits
