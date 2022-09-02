# Configuration File
import os
import sys
import copy

import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()

### build arguments
parser.add_argument('--GNN_Model', default='cPhi_GNN')
parser.add_argument('--save_dir', default='save_dir/optimal', help="Model Save Directory")
parser.add_argument('--eval_dir', default='Numpy', help="Model Save Directory")
parser.add_argument('--output_rate', type=float, default=100, help = 'Rate to Print Learning Progress') ####Fix 500
parser.add_argument('--save_rate', type=float, default=1, help = 'Rate to Save Model')
parser.add_argument('--eval_rate', type=float, default=25, help = 'Rate to Validate Model')
parser.add_argument('--eval_iter', type=int, default=0)
parser.add_argument('--cross_valid_type', default='optimal')
parser.add_argument('--Train_Type', default='train')
parser.add_argument('--learning_rate', type=float, default=0.001, help = 'Learning Rate')
parser.add_argument('--epochs', type=int, default=20, help = 'Number of Epochs')
parser.add_argument('--batch_size', type=int, default=1, help = 'Batches')
parser.add_argument('--OptionIter', default='True')
parser.add_argument('--UsePretrained', default='Nope')
parser.add_argument('--Network_Type', default='Unormalized')
parser.add_argument('--Interpolate_Option', default='Uninterpolate')
parser.add_argument('--Loss_Fn', default='MSE')
parser.add_argument('--verbose', default=False)
parser.add_argument('--visualize', default=False)

parser.add_argument('--nlayers_memory', type=float, default=2)
parser.add_argument('--vertex_filter', type=float, default=150)
parser.add_argument('--edge_filter', type=float, default=150)
parser.add_argument('--vertex_edge_filter', type=float, default=150)
parser.add_argument('--M_steps', type=int, default=10)
parser.add_argument('--nconvolutions', type=int, default=3)

parser.add_argument('--use_multi_aggregators', action='store_true',
                    help='Switch between EGC-S and EGC-M')
parser.add_argument('--n_sims', type=float, default=30) #####Fix 46
parser.add_argument('--n_eval_sims', type=float, default=10) #####Fix 4
parser.add_argument('--n_steps', type=float, default=104)  ####Fix 181
parser.add_argument('--data_len', type=float, default=5)
parser.add_argument('--seq_len', type=float, default=4)
parser.add_argument('--n_coord', type=float, default=2)
parser.add_argument('--feat_dim', type=float, default=9)
parser.add_argument('--out_dim', type=float, default=1)
parser.add_argument('--n_props', type=float, default=2)
parser.add_argument('--dims', type=float, default=193)


def check_dir(PATH):
    if os.path.isdir(PATH):
        pass
    else:
        os.mkdir(PATH)


def args_generator():
    args = parser.parse_args()

    if args.cross_valid_type == 'optimal':
        if args.GNN_Model == 'cPhi_GNN':
            args.learning_rate = 0.0005
            args.nconvolutions = 4
            args.Loss_Fn = 'SmoothL1Loss'
            args.vertex_edge_filter = 0.0
            args.Interpolate_Option = 'Uninterpolate'
            args.Network_Type = 'Unormalized'

        elif args.GNN_Model == 'XDisp_GNN':
            args.learning_rate = 0.001
            args.nconvolutions = 1
            args.Loss_Fn = 'L1Loss'
            args.vertex_edge_filter = 4.0
            args.Interpolate_Option = 'Uninterpolate'
            args.Network_Type = 'Unormalized'

        elif args.GNN_Model == 'YDisp_GNN':
            args.learning_rate = 0.0005
            args.nconvolutions = 6
            args.Loss_Fn = 'MSE'
            args.vertex_edge_filter = 1.0
            args.Interpolate_Option = 'Uninterpolate'
            args.Network_Type = 'Unormalized'
            #args.Network_Type = 'Normalized'

        elif args.GNN_Model == 'SVM_GNN':
            args.learning_rate = 0.0005
            args.nconvolutions = 1
            args.Loss_Fn = 'MSE'
            args.vertex_edge_filter = 1.0
            args.Interpolate_Option = 'Uninterpolate'
            args.Network_Type = 'Unormalized'

    args.edge_index = torch.load('edge_index.pt')
    args.static_node_feats = torch.load('static_node_feats.pt')

    args.my = 0.00013090717
    args.mys = 0.0006539579
    args.mx = -1.792535e-05
    args.mxs = 9.434865e-05

    args.max_svm = 1170600.8
    args.min_svm = 0.9281655
    args.ave_svm = 854.4884

    if args.cross_valid_type != 'plot':
        model_dir = '%s/%s/trained_%s_%s_' % (args.save_dir,args.GNN_Model,args.GNN_Model,args.Network_Type)
        param_dir = '%s%s_%sFilter_' % (model_dir,args.Interpolate_Option,str(args.vertex_edge_filter))
        args.model_main ='%s%sLR_%sConv_%s' % (param_dir,str(args.learning_rate),str(args.nconvolutions),args.Loss_Fn)

        save_1_dir = '%s/%s/Cross_Valid/%s_' % ('results',args.GNN_Model,args.Network_Type)
        save_2_dir = '%s%s_%sFilter_' % (save_1_dir,args.Interpolate_Option,str(args.vertex_edge_filter))
        args.save_case_dir ='%s%sLR_%sConv_%s' % (save_2_dir,str(args.learning_rate),str(args.nconvolutions),args.Loss_Fn)
    

        #check_dir(args.save_case_dir)
    if args.GNN_Model == 'XDisp_GNN':
        args.train_dir = 'train_pt_XDisp/'
    elif args.GNN_Model == 'YDisp_GNN':
        args.train_dir = 'train_pt_YDisp/'
    elif args.GNN_Model == 'cPhi_GNN':
        args.train_dir = 'Numpy/' #'Latest/train_pt_cPhi/'

    if args.Train_Type == 'optimal':
        args.epochs = 35

    return args
