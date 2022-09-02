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
#from torch_geometric.nn.dense.linear import Linear

from typing import Any, Callable, Optional, Union


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import OrderedDict


#from models_HPC import GINEConv
from config_HPC_Latest import args_generator

args = args_generator()


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


#########################################################################
#########################################################################
#########################################################################
def loader(args, SimNO,LoadStep):
    load_str =  args.train_dir + str(SimNO) + '_Case/AllVars_Load_Iter_' + str(LoadStep) + '.npy'
    AllVars = np.load(load_str)

    return AllVars
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def init_mesh(dims):
    prev=0
    test_arr = np.zeros([dims,1])
    test_arr[-1,0] = 1
    for l in range(dims-1):
        prev+=0.002604
        test_arr[l+1,0] = prev

    test_arr = np.round(test_arr,3)
    Mesh_x, Mesh_y = np.meshgrid(test_arr[:,0], test_arr[:,0]) 

    MESH = np.zeros([dims*dims, 2])
    MESH[:,0] = Mesh_x.flatten()
    MESH[:,1] = Mesh_y.flatten()

    return Mesh_x, Mesh_y, MESH
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def get_neighbors(args, AI_MAT, Xs, Ys,SVM_MAT,cPhi_Mat,Xdisp_MAT,Ydisp_MAT,visualize = False):
    tol = 0.001
    n_active = len(Xs)
    nCracks = len(AI_MAT)
    queries_active = np.arange(n_active)
    anchors_active = np.arange(n_active)

    pos = np.zeros([n_active,2])

    for ii in range(n_active):

        pos[ii,0] = Xs[ii]
        pos[ii,1] = Ys[ii]

    tri = Delaunay(pos,incremental=True)

    neighbors = []
    for m in range((n_active)):
        neigh = []
        neigh_refined = []
        neigh.append(m)
        neigh_refined.append(m)
        for mm in range((n_active)):
            

            if (m == tri.simplices[mm,0]) or (m == tri.simplices[mm,1]) or (m == tri.simplices[mm,2]):  
                neigh.append(tri.simplices[mm,0])
                neigh.append(tri.simplices[mm,1])
                neigh.append(tri.simplices[mm,2])
            
        neigh_res = list(OrderedDict.fromkeys(neigh))  

        for pic in range(len(neigh_res)):

            if (Xs[neigh_res[pic]] != Xs[m]) and ((Ys[neigh_res[pic]] != Ys[m])): 
                continue
            else:
                neigh_refined.append(neigh_res[pic])


        neighbors.append(neigh_refined)
    
    e_count = 0
    for lmk in range(len(neighbors)):
        neighbors[lmk] = list(OrderedDict.fromkeys(neighbors[lmk]))  
        e_count+=len(neighbors[lmk])-1
    
    edge_index = torch.zeros([2,e_count], dtype=torch.long)
    edge_index_reference = torch.zeros([2,e_count], dtype=torch.long)
    e_idx_ref=0
    for m in range(len(neighbors)):
        current_n = neighbors[m]
        for f in range(len(neighbors[m])-1):
            edge_index_reference[0,e_idx_ref] = current_n[0]
            edge_index_reference[1,e_idx_ref] = current_n[f+1]
            e_idx_ref+=1

    if visualize:
        check = 0
        for pic in range(len(neighbors)):
            plt.figure()
            plt.scatter(Xs,Ys)
            print(neighbors[pic])
            for l in range(len(neighbors[pic])):
                cur_n = neighbors[pic]
                plt.plot([Xs[pic],Xs[cur_n[l]]], [Ys[pic],Ys[cur_n[l]]], color='red')
                print(edge_index_reference[:,check])
                check+=1
            plt.show()

    Neight=[]
    idd = []
    for i in range(nCracks):
        if AI_MAT[i] != 0:
            idd.append(i) 

    count = 0
    for i in range(nCracks):
        if AI_MAT[i] != 0:
            neigh = neighbors[count]
            cur_p = []
            for k in range(len(neigh)):
                cur_p.append(idd[neigh[k]])

            Neight.append(cur_p) 
            count+=1
            
        elif AI_MAT[i] == 0:
            Neight.append([i])


    e_idx = 0
    for lmk in range(len(Neight)):
        Neight[lmk] = list(OrderedDict.fromkeys(Neight[lmk]))
    
        if len(Neight[lmk]) > 1:
            current_n = Neight[lmk]
            for f in range(len(Neight[lmk])-1):
                edge_index[0,e_idx] = current_n[0]
                edge_index[1,e_idx] = current_n[f+1]
                e_idx+=1
                
    if args.GNN_Model == 'XDisp_GNN' or args.GNN_Model == 'YDisp_GNN':
        edge_attr = torch.zeros((edge_index.shape[1] , int(args.feat_dim-2)))
    else: 
        edge_attr = torch.zeros((edge_index.shape[1] , args.feat_dim))

    for k in range(edge_index.shape[1]):
        sender = edge_index_reference[0,k]
        receiver = edge_index_reference[1,k]
        del_Xs = Xs[receiver] - Xs[sender]
        del_Ys = Ys[receiver] - Ys[sender]
        d = np.sqrt((del_Xs)**2 + (del_Ys)**2)
        edge_attr[k,0] = del_Xs
        edge_attr[k,1] = del_Ys
        edge_attr[k,2] = d 

        if args.GNN_Model == 'XDisp_GNN' or args.GNN_Model == 'YDisp_GNN':
            if del_Xs == 0:
                edge_attr[k,3] = (cPhi_Mat[receiver] - cPhi_Mat[sender])/del_Ys 
            elif del_Ys == 0:
                edge_attr[k,3] = (cPhi_Mat[receiver] - cPhi_Mat[sender])/del_Xs     
            else: 
                edge_attr[k,3] = (cPhi_Mat[receiver] - cPhi_Mat[sender])/del_Xs + (cPhi_Mat[receiver] - cPhi_Mat[sender])/del_Ys
            
            edge_attr[k,4] = (cPhi_Mat[receiver] - cPhi_Mat[sender])

            if del_Xs == 0:
                edge_attr[k,5] = (SVM_MAT[receiver] - SVM_MAT[sender])/del_Ys
            elif del_Ys == 0:
                edge_attr[k,5] = (SVM_MAT[receiver] - SVM_MAT[sender])/del_Xs
            else:
                edge_attr[k,5] = (SVM_MAT[receiver] - SVM_MAT[sender])/del_Xs + (SVM_MAT[receiver] - SVM_MAT[sender])/del_Ys

            edge_attr[k,6] = (SVM_MAT[receiver] - SVM_MAT[sender])
            
        else:
            edge_attr[k,3] = SVM_MAT[receiver] - SVM_MAT[sender] 
            edge_attr[k,4] = cPhi_Mat[receiver] - cPhi_Mat[sender]
            del_xdisp = (Xdisp_MAT[receiver] - Xdisp_MAT[sender])
            del_ydisp = (Ydisp_MAT[receiver] - Ydisp_MAT[sender]) 

            
            del_xdisp_norm = (Xdisp_MAT[receiver] - Xdisp_MAT[sender]) *1000
            del_ydisp_norm = (Ydisp_MAT[receiver] - Ydisp_MAT[sender])*1000/4

            edge_attr[k,5] = del_xdisp_norm 
            edge_attr[k,6] = del_ydisp_norm
            edge_attr[k,7] = np.sqrt( (del_xdisp)**2  + (del_ydisp)**2 )
            edge_attr[k,8] = np.sqrt( (del_xdisp_norm)**2  + (del_ydisp_norm)**2 )



    return edge_index, edge_attr
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def get_active_inactive(args, Mesh_x, Mesh_y, X, Y,SVM,c_Phi,Xdisp,Ydisp, deltaD):
    count = 0
    tol = 0.001
    indexes = []
    Xs = []
    Xn = []
    Ys = []
    Yn = []
    AI_Mat = []
    cPhi_Mat = []
    SVM_MAT = []
    Xdisp_MAT = []
    Ydisp_MAT = []
    deltaD_MAT = []

    for i in range(len(Mesh_x[:,0])):
        for j in range(len(Mesh_y[:,0])):
            val_x = Mesh_x[i,j]
            val_y = Mesh_y[i,j]

            result_x = np.where(((X[:] >= val_x-tol) & (X[:] <= val_x+tol)) & ((Y[:] >= val_y-tol) & (Y[:] <= val_y+tol)))

            is_empty = result_x[0].size == 0
            if is_empty == False:
                indexes.append(result_x[0]) 
                Xs.append(val_x)
                Ys.append(val_y)
                AI_Mat.append(1)
                SVM_MAT.append(SVM[indexes[-1][0]])
                cPhi_Mat.append(c_Phi[indexes[-1][0]])
                Xdisp_MAT.append(Xdisp[indexes[-1][0]])
                Ydisp_MAT.append(Ydisp[indexes[-1][0]])
                deltaD_MAT.append(deltaD[indexes[-1][0]])
                count+=1

            elif is_empty == True:
                Xn.append(val_x)
                Yn.append(val_y)
                AI_Mat.append(0)


    if args.GNN_Model == 'XDisp_GNN' or args.GNN_Model == 'YDisp_GNN':
        node_feats = torch.zeros([len(AI_Mat),int(args.feat_dim-2)])
    else:
        node_feats = torch.zeros([len(AI_Mat),args.feat_dim])

    cc = 0
    nn = 0
    for k in range(len(AI_Mat)):
        if AI_Mat[k] == 1:
            node_feats[k,0] = Xs[cc]
            node_feats[k,1] = Ys[cc]
            node_feats[k,2] = SVM_MAT[cc]/6.0
            node_feats[k,3] = cPhi_Mat[cc]
            if args.Network_Type == 'Normalized':
                node_feats[k,4] = (Xdisp_MAT[cc] - args.mx) / args.mxs
                node_feats[k,5] = (Ydisp_MAT[cc] - args.my) / args.mys
            else:
                node_feats[k,4] = (Xdisp_MAT[cc])*1000 
                node_feats[k,5] = (Ydisp_MAT[cc])*1000/4 
            node_feats[k,6] = deltaD_MAT[cc]
            cc+=1

        elif AI_Mat[k] == 0:
            node_feats[k,0] = Xn[nn]
            node_feats[k,1] = Yn[nn]
            nn+=1


    return Xs, Ys, AI_Mat, SVM_MAT, cPhi_Mat, Xdisp_MAT, Ydisp_MAT, node_feats
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def get_active_output(args,AI_MAT, Xs, Ys,SVM,c_Phi,Xdisp,Ydisp):

    node_output = np.zeros([len(AI_MAT),args.out_dim])
    Node_Output = np.zeros([len(AI_MAT),args.out_dim])
    count = 0
    for k in range(len(AI_MAT)):
        if AI_MAT[k] == 1:        
            if args.GNN_Model == 'cPhi_GNN':
                node_output[k,0] = c_Phi[count]
            elif args.GNN_Model == 'SVM_GNN':
                node_output[k,0] = SVM[count]/6.
            elif args.GNN_Model == 'XDisp_GNN':
                if args.Network_Type == 'Normalized':
                    node_output[k,0] = (Xdisp[count] - args.mx)/args.mxs
                else:
                    node_output[k,0] = Xdisp[count]*1000
            elif args.GNN_Model == 'YDisp_GNN':
                if args.Network_Type == 'Normalized':
                    node_output[k,0] = (Ydisp[count] - args.my)/args.mys
                else:
                    node_output[k,0] = Ydisp[count]*1000/4
            count+=1
        elif AI_MAT[k] == 0:
            node_output[k,0] = 0


    _out = ((np.array(node_output)).reshape(int(args.dims), int(args.dims)))
    _out_n = _out.copy()
    _out_n[_out == 0] = np.average(_out)
    Output = _out_n.flatten()

    Node_Output[:,0] = Output

    return Node_Output
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def input_data(args, X, Y, c_Phi, SVM, Xdisp, Ydisp, deltaD):

    Mesh_x, Mesh_y, MESH = init_mesh(args.dims)

    Xs, Ys, AI_MAT, SVM_MAT, cPhi_Mat, Xdisp_MAT, Ydisp_MAT, node_feats = get_active_inactive(args,Mesh_x, 
    Mesh_y, X, Y, SVM, c_Phi,Xdisp,Ydisp, deltaD)

    return node_feats 
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
def output_data(args, X, Y, c_Phi, SVM, Xdisp, Ydisp, deltaD):

    Mesh_x, Mesh_y, MESH = init_mesh(args.dims)

    Xs, Ys, AI_MAT, SVM_MAT, cPhi_Mat, Xdisp_MAT, Ydisp_MAT, _  = get_active_inactive(args,Mesh_x, 
    Mesh_y, X, Y, SVM, c_Phi,Xdisp,Ydisp, deltaD)

    node_output = get_active_output(args, AI_MAT, Xs, Ys, SVM_MAT, cPhi_Mat, Xdisp_MAT, Ydisp_MAT)
    edge_index, edge_attr = get_neighbors(args, AI_MAT, Xs, Ys,SVM_MAT,cPhi_Mat,Xdisp_MAT,Ydisp_MAT, visualize = False)

    return node_output, edge_index, edge_attr
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################




class MyOwnDataset(Dataset):
    def __init__(self, args):
        super().__init__(args)

        # Define working directory for Simulations Results 
        self.args = args
        
        self.n_steps = args.n_steps
        self.data_len = args.data_len
        self.n_sims = args.n_sims 

    def len(self):
        return (self.n_sims * (self.n_steps - args.seq_len))

    def get(self, idx):
        sim_residual = self.n_steps - self.data_len+1
        idx_sim = idx // sim_residual
        idx_i = idx % sim_residual
        idx_f = idx_i + 1

        NODES, POS, EDGES, TMP = [], [], [], []
        print()
        print('|---- Current Simulation -----> ', idx_sim)
        print()
        init = 0
        for time in range(idx_i,idx_f):
            print('|---- Current Time-step -----> ', time)

            ######################################################################################################
            ######################################################################################################
            ######################################################################################################
            AllVars = loader(args, idx_sim+1, time)
            X = AllVars[:,0]
            Y = AllVars[:,1]
            c_Phi = AllVars[:,2]
            SVM = AllVars[:,3]
            Xdisp = AllVars[:,4]  
            Ydisp = AllVars[:,5]
            deltaD = AllVars[:,6]

            node_feats = input_data(self.args, X, Y, c_Phi, SVM,Xdisp,Ydisp, deltaD)
            NODES.append(node_feats.numpy())

            if args.GNN_Model != 'Disp_GNN':

                AllVars_tmp = loader(args, idx_sim+1, time+1)
                X_tmp = AllVars_tmp[:,0]
                Y_tmp = AllVars_tmp[:,1]
                c_Phi_tmp = AllVars_tmp[:,2]
                SVM_tmp = AllVars_tmp[:,3]
                Xdisp_tmp = AllVars_tmp[:,4]  
                Ydisp_tmp = AllVars_tmp[:,5]
                deltaD_tmp = AllVars_tmp[:,6]

                node_feats_tmp = input_data(self.args, X_tmp, Y_tmp, c_Phi_tmp, SVM_tmp,Xdisp_tmp,Ydisp_tmp, deltaD_tmp)
                TMP.append(node_feats_tmp.numpy())

        ######################################################################################################
        ######################################################################################################
        ######################################################################################################

        AllVars_out = loader(args, idx_sim+1, idx_f)
        X_out = AllVars_out[:,0]
        Y_out = AllVars_out[:,1]
        c_Phi_out = AllVars_out[:,2]
        SVM_out = AllVars_out[:,3]
        Xdisp_out = AllVars_out[:,4]  
        Ydisp_out = AllVars_out[:,5]
        deltaD_out = AllVars_out[:,6]

        node_target, edge_index, edge_attr  = output_data(self.args, X_out, Y_out, c_Phi_out, SVM_out,Xdisp_out,Ydisp_out, deltaD_out)

        b_a = rnn_utils.pad_sequence([ torch.FloatTensor(NODES[0]), torch.FloatTensor(TMP[0]) ])
        a_a = torch.FloatTensor(TMP[0])
        
        feat = b_a.permute(0,2,1)
        Feat = feat[:,:,0]
        if args.GNN_Model == 'cPhi_GNN':
            Feat[:,7] = a_a[:,4] 
            Feat[:,8] = a_a[:,5]


            
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        Node_target = torch.FloatTensor(node_target)

        return Feat, Node_target, edge_index, edge_attr 