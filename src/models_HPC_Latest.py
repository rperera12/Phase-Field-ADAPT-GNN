import copy
import torch
from torch import Tensor
from torch.nn import Linear as Lin
import torch.nn.utils.rnn as rnn_utils
from torch.nn import Sequential as Seq
from torch.utils.data import DataLoader

import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from typing import Any, Callable, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import OrderedDict

from config_HPC_Latest import args_generator

args = args_generator()


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)



class GINEConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = Lin(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'



class Net1(torch.nn.Module):
    def __init__(self, args, dim, cuda_avail):
        super(Net1, self).__init__()

        if args.GNN_Model == 'cPhi_GNN' or args.GNN_Model == 'SVM_GNN':
            self.num_features = args.feat_dim
        else:
            self.num_features = args.feat_dim - 2
        self.dim = dim
        self.cuda_avail = cuda_avail
        
        nn1 = Seq(Lin(self.num_features, dim))
        self.conv1 = GINEConv(nn1, train_eps=True)
        self.lin1 = Lin(dim, dim)
        self.lin2 = Lin(dim, 1)

    def forward(self, x, edge_index, edge_weight):
        
        x = self.conv1(x, edge_index, edge_weight).relu()
        if self.cuda_avail:
            lin_e1 = Lin(self.num_features, self.dim).cuda()
        else:
            lin_e1 = Lin(self.num_features, self.dim)
        x = self.lin1(x).relu()
        out = self.lin2(x)
        return out



class Net2(torch.nn.Module):
    def __init__(self, args, dim, cuda_avail):
        super(Net2, self).__init__()

        if args.GNN_Model == 'cPhi_GNN' or args.GNN_Model == 'SVM_GNN':
            self.num_features = args.feat_dim
        else:
            self.num_features = args.feat_dim - 2
        self.dim = dim
        self.cuda_avail = cuda_avail
        
        nn1 = Seq(Lin(self.num_features, dim))
        self.conv1 = GINEConv(nn1, train_eps=True)
        nn2 = Seq(Lin(dim, dim))
        self.conv2 = GINEConv(nn2, train_eps=True)
        self.lin1 = Lin(dim, dim)
        self.lin2 = Lin(dim, 1)

    def forward(self, x, edge_index, edge_weight):
        
        x = self.conv1(x, edge_index, edge_weight).relu()
        if self.cuda_avail:
            lin_e1 = Lin(self.num_features, self.dim).cuda()
            edge_weight = lin_e1(edge_weight).cuda()
        else:
            lin_e1 = Lin(self.num_features, self.dim)
            edge_weight = lin_e1(edge_weight)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.lin1(x).relu()
        out = self.lin2(x)
        return out


class Net3(torch.nn.Module):
    def __init__(self, args, dim, cuda_avail):
        super(Net3, self).__init__()

        if args.GNN_Model == 'cPhi_GNN' or args.GNN_Model == 'SVM_GNN':
            self.num_features = args.feat_dim
        else:
            self.num_features = args.feat_dim - 2
        self.dim = dim
        self.cuda_avail = cuda_avail
        
        nn1 = Seq(Lin(self.num_features, dim))
        self.conv1 = GINEConv(nn1, train_eps=True)
        nn2 = Seq(Lin(dim, dim))
        self.conv2 = GINEConv(nn2, train_eps=True)
        self.conv3 = GINEConv(nn2, train_eps=True)
        self.lin1 = Lin(dim, dim)
        self.lin2 = Lin(dim, 1)

    def forward(self, x, edge_index, edge_weight):
        
        x = self.conv1(x, edge_index, edge_weight).relu()
        if self.cuda_avail:
            lin_e1 = Lin(self.num_features, self.dim).cuda()
            edge_weight = lin_e1(edge_weight).cuda()
        else:
            lin_e1 = Lin(self.num_features, self.dim)
            edge_weight = lin_e1(edge_weight)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.lin1(x).relu()
        out = self.lin2(x)
        return out


class Net4(torch.nn.Module):
    def __init__(self, args, dim, cuda_avail):
        super(Net4, self).__init__()

        if args.GNN_Model == 'cPhi_GNN' or args.GNN_Model == 'SVM_GNN':
            self.num_features = args.feat_dim
        else:
            self.num_features = args.feat_dim - 2
        self.dim = dim
        self.cuda_avail = cuda_avail
        
        nn1 = Seq(Lin(self.num_features, dim))
        self.conv1 = GINEConv(nn1, train_eps=True)
        nn2 = Seq(Lin(dim, dim))
        self.conv2 = GINEConv(nn2, train_eps=True)
        self.conv3 = GINEConv(nn2, train_eps=True)
        self.conv4 = GINEConv(nn2, train_eps=True)
        self.lin1 = Lin(dim, dim)
        self.lin2 = Lin(dim, 1)

    def forward(self, x, edge_index, edge_weight):
        
        x = self.conv1(x, edge_index, edge_weight).relu()
        if self.cuda_avail:
            lin_e1 = Lin(self.num_features, self.dim).cuda()
            edge_weight = lin_e1(edge_weight).cuda()
        else:
            lin_e1 = Lin(self.num_features, self.dim)
            edge_weight = lin_e1(edge_weight)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.lin1(x).relu()
        out = self.lin2(x)
        return out


class Net5(torch.nn.Module):
    def __init__(self, args, dim, cuda_avail):
        super(Net5, self).__init__()

        if args.GNN_Model == 'cPhi_GNN' or args.GNN_Model == 'SVM_GNN':
            self.num_features = args.feat_dim
        else:
            self.num_features = args.feat_dim - 2
        self.dim = dim
        self.cuda_avail = cuda_avail
        
        nn1 = Seq(Lin(self.num_features, dim))
        self.conv1 = GINEConv(nn1, train_eps=True)
        nn2 = Seq(Lin(dim, dim))
        self.conv2 = GINEConv(nn2, train_eps=True)
        self.conv3 = GINEConv(nn2, train_eps=True)
        self.conv4 = GINEConv(nn2, train_eps=True)
        self.conv5 = GINEConv(nn2, train_eps=True)
        self.lin1 = Lin(dim, dim)
        self.lin2 = Lin(dim, 1)

    def forward(self, x, edge_index, edge_weight):
        
        x = self.conv1(x, edge_index, edge_weight).relu()
        if self.cuda_avail:
            lin_e1 = Lin(self.num_features, self.dim).cuda()
            edge_weight = lin_e1(edge_weight).cuda()
        else:
            lin_e1 = Lin(self.num_features, self.dim)
            edge_weight = lin_e1(edge_weight)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()
        x = self.lin1(x).relu()
        out = self.lin2(x)
        return out


class Net6(torch.nn.Module):
    def __init__(self, args, dim, cuda_avail):
        super(Net6, self).__init__()

        if args.GNN_Model == 'cPhi_GNN' or args.GNN_Model == 'SVM_GNN':
            self.num_features = args.feat_dim
        else:
            self.num_features = args.feat_dim - 2
        self.dim = dim
        self.cuda_avail = cuda_avail
        
        nn1 = Seq(Lin(self.num_features, dim))
        self.conv1 = GINEConv(nn1, train_eps=True)
        nn2 = Seq(Lin(dim, dim))
        self.conv2 = GINEConv(nn2, train_eps=True)
        self.conv3 = GINEConv(nn2, train_eps=True)
        self.conv4 = GINEConv(nn2, train_eps=True)
        self.conv5 = GINEConv(nn2, train_eps=True)
        self.conv6 = GINEConv(nn2, train_eps=True)
        self.lin1 = Lin(dim, dim)
        self.lin2 = Lin(dim, 1)

    def forward(self, x, edge_index, edge_weight):
        
        x = self.conv1(x, edge_index, edge_weight).relu()
        if self.cuda_avail:
            lin_e1 = Lin(self.num_features, self.dim).cuda()
            edge_weight = lin_e1(edge_weight).cuda()
        else:
            lin_e1 = Lin(self.num_features, self.dim)
            edge_weight = lin_e1(edge_weight)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight).relu()
        x = self.conv6(x, edge_index, edge_weight).relu()
        x = self.lin1(x).relu()
        out = self.lin2(x)
        return out
