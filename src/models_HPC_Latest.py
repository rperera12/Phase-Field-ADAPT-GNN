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
import torch
import os
#from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from config_HPC_Latest import args_generator
import numpy as np
import glob
from scipy.spatial import Delaunay
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
#from progressbar import ProgressBar
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
#from ogb.graphproppred import Evaluator
#from ogb.graphproppred import PygGraphPropPredDataset as OGBG
#from ogb.graphproppred.mol_encoder import AtomEncoder
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os.path as osp
import torch.nn.utils.rnn as rnn_utils
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import EGConv, global_mean_pool
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj
from tqdm import tqdm
#from torch_geometric_temporal.signal import temporal_signal_split
import seaborn as sns
from torch_geometric_temporal.nn.recurrent import A3TGCN2 

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
    
    
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h) 
        h = self.linear(h)
        return h


if phase == 'train':
    if args.GNN_Model == 'cPhi_GNN':
        TemporalGNN(node_features=9, periods=1, batch_size=args.batch_size)
    else:
        TemporalGNN(node_features=7, periods=1, batch_size=args.batch_size)
elif phase == 'eval':
    if args.GNN_Model == 'cPhi_GNN':
        TemporalGNN(node_features=9, periods=1, batch_size=1)
    else:
        TemporalGNN(node_features=7, periods=1, batch_size=4)



# Create model and optimizers
if torch.cuda.is_available() == True:
    if phase == 'train':
        if args.GNN_Model == 'cPhi_GNN':
            model = TemporalGNN(node_features=9, periods=1, batch_size=args.batch_size).to(device)
        else:
            model = TemporalGNN(node_features=7, periods=1, batch_size=args.batch_size).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_fn = torch.nn.MSELoss()

        print('Net\'s state_dict:')
        total_param = 0
        for param_tensor in model.state_dict():
            print(param_tensor, '\t', model.state_dict()[param_tensor].size())
            total_param += np.prod(model.state_dict()[param_tensor].size())
        print('Net\'s total params:', total_param)
        #--------------------------------------------------
        print('Optimizer\'s state_dict:')  # If you notice here the Attention is a trainable parameter
        for var_name in optimizer.state_dict():
            print(var_name, '\t', optimizer.state_dict()[var_name])


    elif phase == 'eval':
        if args.GNN_Model == 'cPhi_GNN':
            model = TemporalGNN(node_features=9, periods=1, batch_size=1).to(device)
            pretrained_dict = torch.load('save_dir/trained_cPhiGNN_Angles.pth')
        elif args.GNN_Model == 'XDisp_GNN':
            model = TemporalGNN(node_features=7, periods=1, batch_size=1).to(device)
            pretrained_dict = torch.load('save_dir/trained_XDispGNN_NEW.pth') 
        elif args.GNN_Model == 'YDisp_GNN':
            model = TemporalGNN(node_features=7, periods=1, batch_size=1).to(device)
            pretrained_dict = torch.load('save_dir/trained_YDispGNN_TEST.pth') 
        model.load_state_dict(pretrained_dict, strict=False)
    edge_index = edge_index.to(device)

else:
    if phase == 'train':
        model = TemporalGNN(node_features=9, periods=1, batch_size=4)
    elif phase == 'eval':
        model = TemporalGNN(node_features=9, periods=1, batch_size=1)
