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
parser.add_argument('--train_dir', default='train_pt_cPhi/', 
                    help="training data directory")
parser.add_argument('--eval_dir', default='Numpy', 
                    help="numpy simulations directory")
parser.add_argument('--results_dir', default='results', 
                    help="results directory")
parser.add_argument('--save_dir', default='save_dir', 
                    help="model save directory")
parser.add_argument('--misc_dir', default='misc', 
                    help="miscelaneous files directory")
parser.add_argument('--output_rate', type=float, default=100, 
                    help = 'Rate to Print Learning Progress')
parser.add_argument('--save_rate', type=float, default=1, 
                    help = 'Rate to Save Model')
parser.add_argument('--eval_rate', type=float, default=25, 
                    help = 'Rate to Validate Model')
parser.add_argument('--eval_iter', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.001, 
                    help = 'Learning Rate')
parser.add_argument('--epochs', type=int, default=20, 
                    help = 'Number of Epochs')
parser.add_argument('--batch_size', type=int, default=1, 
                    help = 'Batches')
parser.add_argument('--OptionIter', default='True')
parser.add_argument('--UsePretrained', default='Nope')
parser.add_argument('--verbose', default=False)
parser.add_argument('--visualize', default=False)

parser.add_argument('--vertex_edge_filter', type=float, default=32)
parser.add_argument('--n_sims', type=float, default=30) 
parser.add_argument('--n_eval_sims', type=float, default=10) 
parser.add_argument('--data_len', type=float, default=5)
parser.add_argument('--seq_len', type=float, default=4)
parser.add_argument('--n_coord', type=float, default=2)
parser.add_argument('--feat_dim', type=float, default=9)
parser.add_argument('--out_dim', type=float, default=1)
parser.add_argument('--n_nearest_nodes', type=int, default=10)
parser.add_argument('--dims', type=float, default=193)
parser.add_argument('--domain_width', type=float, default=0.5)


def args_generator():
    args = parser.parse_args()

    if args.GNN_Model == 'XDisp_GNN':
        args.train_dir = 'train_pt_XDisp/'
    elif args.GNN_Model == 'YDisp_GNN':
        args.train_dir = 'train_pt_YDisp/'
    elif args.GNN_Model == 'cPhi_GNN':
        args.train_dir = 'train_pt_cPhi/'

    return args
