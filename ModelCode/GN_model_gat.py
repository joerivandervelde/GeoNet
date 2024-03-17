import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
# from torch_cluster import knn_graph, radius_graph
# from torch_geometric.nn import MetaLayer, voxel_grid, global_max_pool, max_pool, avg_pool
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_batch, to_dense_adj
from torch_scatter import scatter_add, scatter_mean, scatter_max
from .message_passing import MessagePassing


class GATConv(MessagePassing):
    def __init__(self, x_ind, x_hs, u_ind, edge_ind, heads=1, concat=True, negative_slope=0.2, dropout=0.5, bias=True):
        super(GATConv, self).__init__()

        self.x_ind = x_ind
        self.x_hs = x_hs
        self.u_ind = u_ind
        self.edge_ind = edge_ind
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.att_global = nn.Parameter(torch.Tensor(1, self.heads, 2 * x_hs))
        self.pij = nn.Parameter(torch.Tensor(1, self.heads, 3))

        self.edge_model = nn.Sequential(
            nn.Conv1d(2 * x_ind + edge_ind + u_ind, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(x_hs, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs)
        )
        self.edge_mlp = nn.Sequential(
            nn.Conv1d(x_hs + edge_ind, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(x_hs, x_hs, kernel_size=1, bias=True),
            nn.BatchNorm1d(x_hs)
        )

        self.node_model = nn.Sequential(
            nn.Conv1d(x_ind + u_ind + x_hs, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(x_hs, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs)
        )
        self.node_mlp = nn.Sequential(
            nn.Conv1d(x_hs + x_ind, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(x_hs, x_hs, kernel_size=1, bias=True),
            nn.BatchNorm1d(x_hs)
        )

        self.Pij_model = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=1, bias=bias),
            nn.BatchNorm1d(8),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(8, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
        )

        self.global_model = nn.Sequential(
            nn.Conv1d(u_ind + x_hs, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(x_hs, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(x_hs, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs)
        )
        self.global_mlp = nn.Sequential(
            nn.Conv1d(x_hs + u_ind, x_hs, kernel_size=1, bias=bias),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(x_hs, x_hs, kernel_size=1, bias=True),
            nn.BatchNorm1d(x_hs),
        )

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * x_hs))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(x_hs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.pij)
        nn.init.xavier_uniform_(self.att_global)
        for model in zip(
                [self.edge_model, self.edge_mlp, self.node_model, self.node_mlp, self.global_model, self.global_mlp,
                 self.Pij_model]):
            for layer in model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    # nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.Conv1d):
                    nn.init.xavier_uniform_(layer.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, dij, edge_index, edge_attr, global_index, wij, Pij, u, batch, pos):
        x = x.repeat(1, self.heads).view(-1, self.heads, self.x_ind)
        x, u, edge_attr = self.propagate(edge_index, x=x, dij=dij, edge_attr=edge_attr, global_index=global_index,
                                         wij=wij, Pij=Pij, u=u, batch=batch, pos=pos)

        return x, u, edge_attr

    def message(self, edge_index_i, edge_index_j, x, x_i, x_j, edge_attr, dij, global_index, wij, Pij, u, batch,
                pos,
                size_i):
        edge_attr = edge_attr.repeat(1, self.heads).view(-1, self.heads, self.edge_ind)
        u = u.repeat(1, self.heads).view(-1, self.heads, self.u_ind)

        dij = dij.repeat(1, self.heads).view(-1, self.heads, 1)

        edge_attr_out = torch.cat([x_i, x_j, edge_attr * dij, u[batch][edge_index_j]], dim=-1)
        edge_attr_out = self.edge_model(edge_attr_out.permute(1, 2, 0)).permute(2, 0, 1)
        edge_attr = self.edge_mlp(torch.cat([edge_attr, edge_attr_out], dim=-1).permute(1, 2, 0)).permute(2, 0, 1)

        x_out = torch.cat([x, u[batch], scatter_add(edge_attr * dij, edge_index_j, dim=0, dim_size=x.size(0))], dim=-1)
        x_out = self.node_model(x_out.permute(1, 2, 0)).permute(2, 0, 1)
        x = self.node_mlp(torch.cat([x, x_out], dim=-1).permute(1, 2, 0)).permute(2, 0, 1)

        att = (torch.cat([x[global_index[0]], x[global_index[1]]], dim=-1) * self.att_global).sum(dim=-1)
        att = F.leaky_relu(att, self.negative_slope)
        att = softmax(att, global_index[0], num_nodes=size_i)
        # att = F.dropout(att, p=self.dropout, training=self.training)
        pij_w = (Pij[:, None, :].repeat(1, self.heads, 1) * self.pij).sum(dim=-1)
        pij_w = F.leaky_relu(pij_w, self.negative_slope)
        pij_w = softmax(pij_w, global_index[0], num_nodes=size_i)
        # pij_w = F.dropout(pij_w, p=self.dropout, training=self.training)
        wij = wij.repeat(1, self.heads)
        wij = torch.cat([att, wij, pij_w], dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1).repeat(1, self.heads, 1)
        # wij = wij.repeat(1, self.heads).unsqueeze(-1)

        Pij = self.Pij_model(Pij[:, None, :].repeat(1, self.heads, 1).permute(1, 2, 0)).permute(2, 0, 1)
        # u_out = scatter_add(x * wij, batch, dim=0)
        u_out = scatter_add(wij * Pij * x, batch, dim=0)
        u_out = torch.cat([u, u_out], dim=-1)
        u_out = self.global_model(u_out.permute(1, 2, 0)).permute(2, 0, 1)
        u = self.global_mlp(torch.cat([u, u_out], dim=-1).permute(1, 2, 0)).permute(2, 0, 1)

        return x.mean(dim=1), u.mean(dim=1), edge_attr.mean(dim=1)
        # return x.max(dim=1)[0], u.max(dim=1)[0], edge_attr.max(dim=1)[0]  #bad


class GAT(nn.Module):
    def __init__(self, x_ind, x_hs, u_ind, edge_ind, heads=1, nlayers=4, concat=False,
                 negative_slope=0.2, dropout=0.5, bias=False):
        super(GAT, self).__init__()

        self.nlayers = nlayers
        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(
            GATConv(x_ind, x_hs, u_ind, edge_ind, heads, concat, negative_slope, dropout, bias))

        for i in range(1, nlayers):
            self.gat_layers.append(
                GATConv(x_hs, x_hs, x_hs, x_hs, heads, concat, negative_slope, dropout, bias))

        self.clf = nn.Sequential(
            # nn.Linear((x_hs * nlayers) * 2 + x_ind + u_ind, x_hs),
            nn.Linear((x_hs * nlayers) * 2, x_hs),
            nn.BatchNorm1d(x_hs),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(x_hs, 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.clf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def forward(self, x, dij, edge_index, edge_attr, global_index, wij, Pij, u, batch, pos):
        u_0 = u
        # x_batch, padding = to_dense_batch(x=x, batch=batch, fill_value=0)
        # x_self_output, global_output = [x_batch[:, 0, :]], [u_0]
        x_self_output, global_output = [], []
        for i in range(self.nlayers):
            x, u, edge_attr = self.gat_layers[i](x, dij, edge_index, edge_attr, global_index, wij, Pij, u, batch, pos)
            x_batch, padding = to_dense_batch(x=x, batch=batch, fill_value=0)
            x_self_output.append(x_batch[:, 0, :])
            global_output.append(u)

        x_self_output = torch.cat(x_self_output, dim=1)
        global_output = torch.cat(global_output, dim=1)

        out = torch.cat([x_self_output, global_output], dim=1)
        out = self.clf(out)
        out = F.softmax(out, -1)
        return out[:, -1]


class GeoNet(torch.nn.Module):
    def __init__(self, nlayers, heads, x_ind, edge_ind, x_hs, e_hs, u_hs, dropratio, bias, edge_method, r_list,
                 edge_aggr, node_aggr, dist, max_nn, apply_edgeattr, apply_nodeposemb):
        super(GeoNet, self).__init__()
        self.dist = dist
        self.max_nn = max_nn
        self.u_ind = 3
        self.edge_method = edge_method
        if apply_nodeposemb is False:
            x_ind -= 1

        self.bn = nn.ModuleList([nn.BatchNorm1d(x_ind),
                                 nn.BatchNorm1d(2),
                                 nn.BatchNorm1d(3),
                                 nn.BatchNorm1d(3),
                                 ])

        self.r_list = r_list
        self.apply_edgeattr = apply_edgeattr
        self.apply_nodeposemb = apply_nodeposemb
        self.heads = heads

        self.gat = GAT(x_ind=x_ind, x_hs=x_hs, u_ind=self.u_ind, edge_ind=edge_ind, nlayers=nlayers,
                       dropout=dropratio, heads=self.heads, bias=bias)

    def forward(self, data):
        x, pos, dij, batch, u_0 = data.x, data.pos, data.dij, data.batch, data.u_0
        u_0 = self.bn[2](u_0.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)

        global_index, wij, Pij = data.global_index, data.wij, data.Pij

        radius_index_list, radius_attr_list = data.edge_index, data.edge_attr
        radius_attr_list = self.bn[1](radius_attr_list.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)

        if self.apply_nodeposemb is True:
            x = torch.cat([x, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)
        x = self.bn[0](x.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        Pij = self.bn[3](Pij.permute(1, 0).unsqueeze(0)).squeeze().permute(1, 0)
        out = self.gat(x, dij, radius_index_list, radius_attr_list, global_index, wij, Pij, u_0, batch, pos)

        return out
