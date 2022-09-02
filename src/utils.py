import os
import glob
import math
import scipy.io as io
import scipy.spatial as spatial
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from config_HPC_Latest import args_generator


args = args_generator()

def check_dir(PATH):
    if os.path.isdir(PATH):
        pass
    else:
        os.mkdir(PATH)

#########################################################################
#########################################################################
#########################################################################
def my_collate(batch):
    len_batch = len(batch[0])
    len_rel = 2

    ret = []
    for i in range(len_batch - len_rel):
        d = [(item[i]) for item in batch]
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)

    # processing relations
    # R: B x seq_length x n_rel x (n_p + n_s)
    for i in range(len_rel):
        R = [item[-len_rel + i] for item in batch]
        max_n_rel = 0
        seq_length, _, N = R[0].size()
        for j in range(len(R)):
            max_n_rel = max(max_n_rel, R[j].size(1))
        for j in range(len(R)):
            r = R[j]
            zeros_test = torch.zeros(seq_length,max_n_rel - r.size(1), N)
            r = torch.cat([r, zeros_test], 1)
            R[j] = r

        R = torch.FloatTensor(torch.stack(R))

        ret.append(R)

    return tuple(ret)



#########################################################################
#########################################################################
#########################################################################
def get_temp(SimNO, LoadStep):
    str_load = args.train_dir + 'Matlab/' + str(SimNO) + '_Case/Loadstep_' + str(LoadStep+1) +'_SaveVariables.mat'
    temp = scipy.io.loadmat(str_load)    
    
    return temp, str_load

def get_force(SimNO):
    # Text file data converted to integer data type
    File_str =  args.train_dir + str(SimNO) + '_Case/FD-tensile.txt'
    File_data = np.loadtxt(File_str) #, dtype=int)

    return File_data  



def loader(SimNO,LoadStep):
    load_str =  args.train_dir + str(SimNO) + '_Case/AllVars_Load_Iter_' + str(LoadStep) + '.npy'
    AllVars = np.load(load_str)

    return AllVars


#########################################################################
#########################################################################
#########################################################################
def get_len_idx_arr(total_sims,OptionIter=False):
    total_len = 0
    for eval_iter in range(total_sims):
        
        if OptionIter == True:
            eval_case = args.train_dir + str(eval_iter+1) + '_Case/'
        elif OptionIter == False:
            eval_case = 'train_skipIter/' + str(eval_iter+1) + '_Case/'

        matFiles = len(glob.glob1(eval_case,"*.mat"))

        sim_len = (len(os.listdir(eval_case)) - matFiles - args.data_len - 1) 
        total_len += sim_len

    sim_idx = 0
    track = 0
    idx_arr = np.zeros([total_len,3])
    for eval_iter in range(total_sims):

        if OptionIter == True:
            eval_case = args.train_dir + str(eval_iter+1) + '_Case/'
        elif OptionIter == False:
            eval_case = 'train_skipIter/' + str(eval_iter+1) + '_Case/'
        matFiles = len(glob.glob1(eval_case,"*.mat"))
        sim_len = (len(os.listdir(eval_case)) - matFiles - args.data_len - 1) 
        
        for idx in range(sim_len):
            idx_arr[track,0] = int(track)
            idx_arr[track,1] = int(eval_iter)
            idx_arr[track,2] = int(idx)
            track+=1

    if OptionIter == True:
        np.save('len_tracker.npy', idx_arr)
    elif OptionIter == False:
        np.save('len_tracker_skipIter.npy', idx_arr)


#########################################################################
#########################################################################
#########################################################################
def save_files(SimNO,MaxLoadStep,log=False, IterOption=False):
    count = 0
    fcount = 0
    force = []
    eval_case = args.train_dir + 'Matlab/'+ str(SimNO) + '_Case/'
    matFiles = len(glob.glob1(eval_case,"*.mat")) - 1
    for load_step in range(matFiles):
        temp, str_load = get_temp(SimNO,load_step)
        iters = len(temp['SaveVar']['X'][0,:])

        if iters > 1:
            if IterOption == True:
                real_iter = iters 

                for miter in range(real_iter):    
                    X = temp['SaveVar']['X'][0,miter] 
                    Y = temp['SaveVar']['Y'][0,miter]
                    Xdisp = temp['SaveVar']['Xdisp'][0,miter]
                    Ydisp = temp['SaveVar']['Ydisp'][0,miter]
                    CrackField = temp['SaveVar']['Wcoord'][0,miter]                 
                    SVM = temp['SaveVar']['SVM'][0,miter]
                    if log:
                        SVM = np.log10(SVM)

                    X = np.round(X,3)
                    Y = np.round(Y,3)

                    AllVars = np.zeros([len(X[:,0]), 6])
                    AllVars[:,0] = X[:,0]
                    AllVars[:,1] = Y[:,0]
                    AllVars[:,2] = CrackField[:,0]
                    AllVars[:,3] = SVM[:,0]
                    AllVars[:,4] = Xdisp[:,0]
                    AllVars[:,5] = Ydisp[:,0]
                    
                    #save_str = args.train_dir + str(SimNO) + '_Case/AllVars_Load_Iter_' + str(count) + '.npy'
                    save_str_prev = args.train_dir + 'Numpy/' + str(SimNO) + '_Case/'
                    check_dir(save_str_prev) 
                    save_str = save_str_prev + 'AllVars_Load_Iter_' + str(count) + '.npy'
                    np.save(save_str, AllVars)
                    
                    count+=1

            elif IterOption == False:
                X = temp['SaveVar']['X'][0,-1] 
                Y = temp['SaveVar']['Y'][0,-1]
                Xdisp = temp['SaveVar']['Xdisp'][0,-1]
                Ydisp = temp['SaveVar']['Ydisp'][0,-1]
                CrackField = temp['SaveVar']['Wcoord'][0,-1]                 
                SVM = temp['SaveVar']['SVM'][0,-1]
                if log:
                    SVM = np.log10(SVM)

                X = np.round(X,3)
                Y = np.round(Y,3)

                AllVars = np.zeros([len(X[:,0]), 6])
                AllVars[:,0] = X[:,0]
                AllVars[:,1] = Y[:,0]
                AllVars[:,2] = CrackField[:,0]
                AllVars[:,3] = SVM[:,0]
                AllVars[:,4] = Xdisp[:,0]
                AllVars[:,5] = Ydisp[:,0]
                
                save_str = 'train_skipIter/' + str(SimNO) + '_Case/AllVars_Load_Iter_' + str(count) + '.npy'
                np.save(save_str, AllVars)

                count+=1

        else:
            
            X = scipy.io.loadmat(str_load, variable_names = ("SaveVar",) )["SaveVar"]["X"]
            Y = scipy.io.loadmat(str_load, variable_names = ("SaveVar",) )["SaveVar"]["Y"]
            CrackField = scipy.io.loadmat(str_load, variable_names = ("SaveVar",) )["SaveVar"]['Wcoord'] 
            SVM = scipy.io.loadmat(str_load, variable_names = ("SaveVar",) )["SaveVar"]['SVM']
            Xdisp = scipy.io.loadmat(str_load, variable_names = ("SaveVar",) )["SaveVar"]['Xdisp']
            Ydisp = scipy.io.loadmat(str_load, variable_names = ("SaveVar",) )["SaveVar"]['Ydisp']
            X = X[0,0]
            Y = Y[0,0]
            Xdisp = Xdisp[0,0]
            Ydisp = Ydisp[0,0]
            CrackField = CrackField[0,0]
            SVM = SVM[0,0]
            if log:
                SVM = np.log10(SVM)

            X = np.round(X,3)
            Y = np.round(Y,3)

            AllVars = np.zeros([len(X[:,0]), 6])
            AllVars[:,0] = X[:,0]
            AllVars[:,1] = Y[:,0]
            AllVars[:,2] = CrackField[:,0]
            AllVars[:,3] = SVM[:,0]
            AllVars[:,4] = Xdisp[:,0]
            AllVars[:,5] = Ydisp[:,0]

            if IterOption == True:
                save_str_prev = args.train_dir + 'Numpy/' + str(SimNO) + '_Case/'
                check_dir(save_str_prev) 
                save_str = save_str_prev + 'AllVars_Load_Iter_' + str(count) + '.npy'

            elif IterOption == False:
                save_str = 'train_skipIter/' + str(SimNO) + '_Case/AllVars_Load_Iter_' + str(count) + '.npy'
            np.save(save_str, AllVars)
                
            count+=1




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


def get_active_inactive(Mesh_x, Mesh_y, X, Y,SVM,c_Phi):
    count = 0
    tol = 0.001
    indexes = []
    Xs = []
    Ys = []
    Xn = []
    Yn = []
    AI_Mat = []
    cPhi_Mat = []
    SVM_MAT = []

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
                count+=1
            elif is_empty == True:
                Xn.append(val_x)
                Yn.append(val_y)
                AI_Mat.append(0)

    new_count = 0
    new_arr = []
    for lmk in range(count):
        tmp_indexes = indexes[lmk]
        new_count+=len(tmp_indexes)
        for wn in range(len(tmp_indexes)):
            new_arr.append(tmp_indexes[wn])

    return Xs, Ys, Xn, Yn, AI_Mat, SVM_MAT, cPhi_Mat, count
 
#########################################################################
#########################################################################
#########################################################################
def acc_metric(predb, yb):       
    return (predb.argmax(dim=1) == yb).float().mean()

    
def get_refinement_force(SimNO,idx):
    # Text file data converted to integer data type
    File_str =  args.train_dir + str(SimNO) + '_Case/FD-tensile.txt'
    File_data = np.loadtxt(File_str) 
    load = File_data[idx+1,:]

    return load


def get_active_refinement_input(SimNO, idx, Mesh_x, Mesh_y, X, Y,SVM,c_Phi):
    count = 0
    tol = 0.001
    indexes, Xs, Ys, AI_Mat, cPhi_Mat, SVM_MAT, Load1, Load2 = [], [], [], [], [], [], [], []
    loads = get_refinement_force(SimNO, idx)
    
    for i in range(len(Mesh_x[:,0])-1):
        for j in range(len(Mesh_y[:,0])-1):
            val_x = Mesh_x[i,j+1]
            val_y = Mesh_y[i+1,j]

            result_x = np.where(((X[:] >= val_x-tol) & (X[:] <= val_x+tol)) & ((Y[:] >= val_y-tol) & (Y[:] <= val_y+tol)))

            is_empty = result_x[0].size == 0
            if is_empty == False:
                indexes.append(result_x[0]) 
                Xs.append(val_x)
                Ys.append(val_y)
                AI_Mat.append(1)
                SVM_MAT.append(SVM[indexes[-1][0]])
                cPhi_Mat.append(c_Phi[indexes[-1][0]])
                Load1.append(loads[0])
                Load2.append(loads[1])
                count+=1
            elif is_empty == True:
                Xs.append(val_x)
                Ys.append(val_y)
                SVM_MAT.append(0)
                AI_Mat.append(0)
                cPhi_Mat.append(0)
                Load1.append(0)
                Load2.append(0)

    new_count = 0
    new_arr = []
    for lmk in range(count):
        tmp_indexes = indexes[lmk]
        new_count+=len(tmp_indexes)
        for wn in range(len(tmp_indexes)):
            new_arr.append(tmp_indexes[wn])

    train_in = torch.zeros(7,len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1)
    train_in[0,:,:] = torch.FloatTensor((np.array(Xs[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))
    train_in[1,:,:] = torch.FloatTensor((np.array(Ys[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))
    train_in[2,:,:] = torch.FloatTensor((np.array(AI_Mat[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))
    train_in[3,:,:] = torch.FloatTensor((np.array(SVM_MAT[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))
    train_in[4,:,:] = torch.FloatTensor((np.array(cPhi_Mat[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))
    train_in[5,:,:] = torch.FloatTensor((np.array(Load1[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))
    train_in[6,:,:] = torch.FloatTensor((np.array(Load2[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))

    train_out = torch.zeros(1,len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1)
    train_out[0,:,:] = torch.FloatTensor((np.array(AI_Mat[:])).reshape(len(Mesh_x[:,0])-1, len(Mesh_x[:,0])-1))


    return train_in, train_out



#########################################################################
#########################################################################
#########################################################################
def get_active_refinement_input_simple(SimNO, idx, Mesh_x, Mesh_y, X, Y,SVM,c_Phi):
    tol = 0.001
    AI_Mat, cPhi_Mat, Xs, Ys, indexes,SVM_MAT, Load1, Load2 = [], [], [], [], [], [], [], []
    
    for i in range(len(Mesh_x[:,0])):
        for j in range(len(Mesh_y[:,0])):
            val_x = Mesh_x[i,j]
            val_y = Mesh_y[i,j]

            result_x = np.where(((X[:] >= val_x-tol) & (X[:] <= val_x+tol)) & ((Y[:] >= val_y-tol) & (Y[:] <= val_y+tol)))

            is_empty = result_x[0].size == 0
            if is_empty == False:
                indexes.append(result_x[0]) 
                AI_Mat.append(0)
                cPhi_Mat.append(c_Phi[indexes[-1][0]])
            elif is_empty == True:
                AI_Mat.append(1)
                cPhi_Mat.append(-1)

    train_in = torch.zeros(2,len(Mesh_x[:,0]), len(Mesh_x[:,0]))
    train_in[0,:,:] = torch.FloatTensor((np.array(AI_Mat[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[1,:,:] = torch.FloatTensor((np.array(cPhi_Mat[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))

    train_out = torch.zeros(1,len(Mesh_x[:,0]), len(Mesh_x[:,0]))
    train_out[0,:,:] = torch.FloatTensor((np.array(AI_Mat[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))

    return train_in, train_out

#########################################################################
#########################################################################
#########################################################################
def get_active_displacement_input(SimNO, idx, Mesh_x, Mesh_y, X, Y,SVM,c_Phi,Xdisp,Ydisp):
    count = 0
    tol = 0.001
    indexes, Xs, Ys, AI_Mat, cPhi_Mat, SVM_MAT, Load1, Load2, Xsdisp,Ysdisp = [], [], [], [], [], [], [], [],[],[]
    loads = get_refinement_force(SimNO, idx)
    
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
                Xsdisp.append(Xdisp[indexes[-1][0]])
                Ysdisp.append(Ydisp[indexes[-1][0]])
                Load1.append(loads[0])
                Load2.append(loads[1])
                count+=1
            elif is_empty == True:
                Xs.append(val_x)
                Ys.append(val_y)
                SVM_MAT.append(0)
                AI_Mat.append(0)
                cPhi_Mat.append(-1)
                Xsdisp.append(0)
                Ysdisp.append(0)
                Load1.append(0)
                Load2.append(0)

    new_count = 0
    new_arr = []
    for lmk in range(count):
        tmp_indexes = indexes[lmk]
        new_count+=len(tmp_indexes)
        for wn in range(len(tmp_indexes)):
            new_arr.append(tmp_indexes[wn])

    train_in = torch.zeros(9,len(Mesh_x[:,0]), len(Mesh_x[:,0]))
    train_in[0,:,:] = torch.FloatTensor((np.array(Xs[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[1,:,:] = torch.FloatTensor((np.array(Ys[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[2,:,:] = torch.FloatTensor((np.array(AI_Mat[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[3,:,:] = torch.FloatTensor((np.array(SVM_MAT[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[4,:,:] = torch.FloatTensor((np.array(cPhi_Mat[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[5,:,:] = torch.FloatTensor((np.array(Load1[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[6,:,:] = torch.FloatTensor((np.array(Load2[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[7,:,:] = torch.FloatTensor((np.array(Xsdisp[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_in[8,:,:] = torch.FloatTensor((np.array(Ysdisp[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))

    train_out = torch.zeros(2,len(Mesh_x[:,0]), len(Mesh_x[:,0]))
    train_out[0,:,:] = torch.FloatTensor((np.array(Xsdisp[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))
    train_out[1,:,:] = torch.FloatTensor((np.array(Ysdisp[:])).reshape(len(Mesh_x[:,0]), len(Mesh_x[:,0])))

    return train_in, train_out


#########################################################################
#########################################################################
#########################################################################
def plot_mesh_active(Xs, Ys, X, Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18,12))
    title_str = 'Loadstep: ' + str(50)
    fig.suptitle(title_str)
    ax1.scatter(Xs,Ys)
    ax2.scatter(X[:],Y[:])
    plt.show()

def plot_active_inactive(Xs, Ys, Xn, Yn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18,12))
    ax1.scatter(Xs,Ys)
    ax2.scatter(Xn,Yn, s=0.1, color='red')
    ax2.scatter(Xs,Ys, s=5.0, color='blue')
    plt.show()

#########################################################################
#########################################################################
#########################################################################
def get_neighbors(AI_MAT, Xs, Ys, neighbor_radius):
    nCracks = len(AI_MAT[:])
    n_active = len(Xs)
    queries_active = np.arange(n_active)
    anchors_active = np.arange(n_active)

    pos = np.zeros([n_active,2])

    for ii in range(n_active):
        #if AI_MAT[ii] == 1:

        pos[ii,0] = Xs[ii]
        pos[ii,1] = Ys[ii]

    point_tree = spatial.cKDTree(pos[anchors_active])
    neighbors = point_tree.query_ball_point(pos[queries_active], neighbor_radius, p=2)

    rels = []
    min_neighbors = None
    for i in range(len(neighbors)):
        if min_neighbors is None:
            min_neighbors = len(neighbors[i])
        elif len(neighbors[i]) < min_neighbors:
            min_neighbors = len(neighbors[i])
        else:
            pass

    for i in range(len(neighbors)):
        receiver = np.ones(min_neighbors, dtype=np.int) * queries_active[i]
        sender = np.array(anchors_active[neighbors[i][:min_neighbors]])
        rels.append(np.stack([receiver, sender], axis=1))
    
    
    if len(rels) > 0:
        rels = np.concatenate(rels, 0)

    n_rel = rels.shape[0]
    Rr = torch.zeros(n_rel, n_active)
    Rs = torch.zeros(n_rel, n_active)
    Rr[np.arange(n_rel), rels[:, 0]] = 1
    Rs[np.arange(n_rel), rels[:, 1]] = 1

    return Rr, Rs   



#########################################################################
#########################################################################
#########################################################################
def neighboring_nodes(neighbors,Xs):

    nCracks = len(Xs[:])
    c_neighbor_r = np.array([])
    c_neighbor_s = np.array([])

    for ii in range(nCracks):

        cur_crack = neighbors[ii]
        # Define the obvious rigidity for same-crack-nodes
        if len(cur_crack) == 0:
            
            c_neighbor_r = np.append(c_neighbor_r, (ii))
            c_neighbor_s = np.append(c_neighbor_s, (ii))

        elif len(cur_crack) != 0:
            
            # Take care of the first Node 
            c_neighbor_r = np.append(c_neighbor_r, (ii))
            
            for k in range(len(cur_crack)):
                if (cur_crack[k] != ii):
                
                    
                    c_neighbor_r = np.append(c_neighbor_r, (ii))
                    c_neighbor_s = np.append(c_neighbor_s, (cur_crack[k]))
                

    print('The shape of c_neighbor is ', c_neighbor_r.shape)
    return c_neighbor_r, c_neighbor_s


#########################################################################
#########################################################################
#########################################################################
def create_relations(receiver, sender, Xs):
    nCracks = len(Xs[:])

    get_index_range = np.zeros([nCracks,2])
    relations = []
    for ii in range(nCracks):

        all_index_1 = np.array([])
        
        for lmn in range(len(receiver)):
            if receiver[lmn] == ii:
                all_index_1 = np.append(all_index_1, lmn)

        
        get_index_range[ii,0] = np.min(all_index_1)+1 #Delete the first row for repeated relation
        get_index_range[ii,1] = np.max(all_index_1)


        receiver_1 = receiver[(int(get_index_range[ii,0])):(int(get_index_range[ii,1])+1)]
        sender_1 = sender[(int(get_index_range[ii,0])):(int(get_index_range[ii,1])+1)]

        print(receiver_1.shape)
        print(sender_1.shape)
        
        relations.append(np.stack([receiver_1,sender_1], axis=1))

    return relations




#########################################################################
#########################################################################
#########################################################################
def Rotation_Matrices(relations, Xs):
    nCracks = len(Xs[:])

    if len(relations) > 0:
        rels = np.concatenate(relations, 0)

    count_rels = rels.shape[0]
    
    Rr = torch.zeros(count_rels, nCracks*2) 
    Rs = torch.zeros(count_rels, nCracks*2)  
    Rr[np.arange(count_rels), rels[:,0]] = 1
    Rs[np.arange(count_rels), rels[:,1]] = 1   

    return Rr, Rs



class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_gradient(step):
    def hook(grad):
        print(step, torch.mean(grad, 1)[:4])
    return hook


def add_log(fn, content, is_append=True):
    if is_append:
        with open(fn, "a+") as f:
            f.write(content)
    else:
        with open(fn, "w+") as f:
            f.write(content)


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)   



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

if __name__ == "__main__":
    lightblue = (178/256, 210/256, 210/256)
    greenishyellow = (235/256, 231/256, 170/256)
    lightgreen = (228/256, 241/256, 210/256)
    bage = (252/256, 241/256, 197/256)
    olive = (192/256, 223/256, 149/256)
    lightgray = (234/256, 234/256, 234/256)
    darkerlightblue = (181/256, 219/256, 251/256)
    opaqueblue = (81/256, 113/256, 191/256)
    violet = (223/256, 222/256, 251/256)
    rojo = (234/256, 52/256, 36/256)
    opaqueorange = (245/256, 201/256, 158/256)
    darkerorange = (198/256, 101/256, 38/256)
    
    sims=np.arange(0,12,1)
    work_dir = os.getcwd()
    
    
    matc1 = io.loadmat('results/0.2_Workspace.mat')['c']
    matc2 = io.loadmat('results/0.2_Workspace.mat')['c2']
    print(matc1)
    print(matc2)
    
    #mat = io.loadmat('results/0.1_Workspace.mat')
    for sim in sims:
        print('\n\nWorking on Simulation #%a' % sim)
        mat2 = io.loadmat('results/sim_%a_time.mat' % sim)['TIME']
    
        print(mat2)
        
        sim_time = (mat2[0,0]*60*60 + mat2[1,0]*60 + mat2[2,0])/60
        print(sim_time)
        
        

    PhaseField = (0.425458056,	
    0.412671192,	
    0.437848052	,
    0.454422602	,
    0.708428349	,
    0.456139736	,
    0.425089238	,
    0.433789675	,
    0.505192612	,
    0.432752044	,
    0.448469231	,
    0.3784384)	

    GNN = (0.029348323,
    0.023804962,
    0.01696823,
    0.015091481,
    0.049074707,
    0.02976109,
    0.017586773,
    0.025707364,
    0.014192884,
    0.014402008,
    0.040708813,
    0.042174019)
    
    for k in range(len(GNN)):
        print(PhaseField[k]/GNN[k])
    
    fig, ax = plt.subplots(figsize=(19.5,16), facecolor='white')
    axis_font_bar = {'fontname':'Arial', 'size':'44'}
    ax.set_facecolor('white') #lightgreen)

    n_ticks_bar = np.array([1,3])
    data1 = np.average(PhaseField)
    data1_err = np.std(PhaseField)
    data2 = np.average(GNN)
    data2_err = np.std(GNN)
    
    print((data1+data1_err) / (data2-data2_err) )

    plt.bar(n_ticks_bar[0], data1, yerr=data1_err, align='center', capsize=10, 
                color =lightblue )
    plt.bar(n_ticks_bar[1], data2, yerr=data2_err, align='center', capsize=10, 
                color=bage )

    my_xticks = ['Phase field', 'GNN']
    plt.xticks(n_ticks_bar, my_xticks,fontsize=40)
    plt.yticks(np.round((np.arange(0,data1+data1_err+(data1+data1_err)/10,
                    (data1+data1_err)/10)),2),fontsize=40)
    #plt.ylabel(r'Simulation time $(\frac{min}{timestep})$', **axis_font_bar)

    plt.savefig('Simulation_Time.png', dpi = 300)
    plt.close('all')
    
    



