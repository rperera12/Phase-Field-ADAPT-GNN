#########################################################################################
#########################################################################################
#########################################################################################
import os
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric_temporal.nn.recurrent import A3TGCN2 

from configuration import args_generator
from utils import get_idxi_idxf, label_interpolation, encoder_interpolation

#########################################################################################
#########################################################################################
#########################################################################################
args = args_generator()

def check_dir(PATH):
    if os.path.isdir(PATH):
        pass
    else:
        os.mkdir(PATH)

phase = 'train'
results_dir = 'results/%s/%a_Case' % (args.GNN_Model,args.eval_iter)

#########################################################################################
#########################################################################################
#########################################################################################
def loader(args, SimNO,LoadStep):
    load_str =  ('new_test_Disp/' + str(SimNO) + 
                 '_Case/AllVars_Load_Iter_' + str(LoadStep) + '.npy')
    AllVars = np.load(load_str)

    return AllVars

#########################################################################################
#########################################################################################
#########################################################################################
class MyOwnDataset(Dataset):
    def __init__(self, args):
        super().__init__(args)

        # Define working directory for Simulations Results 
        self.args = args
        self.data_len = args.data_len
        self.train_dir = args.train_dir

    def len(self):
        return 5670

    def get(self, idx):
        idx_sim, idx_i = get_idxi_idxf(idx)
        idx_f = idx_i+4

        NODES = []

        for time  in range(idx_i,idx_f):
            
            sim_dir = f'{self.train_dir}{int(idx_sim+1)}_Case/iter_{int(time)}_' 
            t_str = sim_dir + args.GNN_Model + '_Unormalized_Uninterpolate_'

            node_feat_str = sim_dir + 'input.pt'
            node_target_str = t_str + 'target.pt' 

            Feat = torch.load(node_feat_str)
            Node_target = torch.load(node_target_str)
            NODES.append(Feat.numpy())

        Node_Feats = torch.FloatTensor(np.array(NODES))
        node_feats = Node_Feats[:,0,:,:].permute(1,2,0)
        node_target = torch.zeros(Node_target.shape[1], 1)
        Node_Target = Node_target[0,:,0]
        node_target[:,0] = Node_Target

        return node_feats, node_target

#########################################################################################
#########################################################################################
#########################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = MyOwnDataset(args) 
train_dl = DataLoader(
    data,
    batch_size=args.batch_size,
    shuffle=True) 
print('Length of current dataset: ', len(train_dl))

edge_index_ = torch.load('edge_index.pt')
edge_index = edge_index_.to(device)

#########################################################################################
#########################################################################################
#########################################################################################
# Making the model 
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()

        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, 
                            periods=periods,batch_size=batch_size) 

        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index) 
        h = F.relu(h) 
        h = self.linear(h)
        return h

TemporalGNN(node_features=10, periods=1, batch_size=args.batch_size)

# Create model and optimizers
if torch.cuda.is_available() == True:

    model = TemporalGNN(node_features=10, periods=1, 
                        batch_size=args.batch_size).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = torch.nn.MSELoss()

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)
    #--------------------------------------------------
    print('Optimizer\'s state_dict:')  
    # If you notice here the Attention is a trainable parameter
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

else:
    model = TemporalGNN(node_features=10, periods=1, batch_size=4)
    

#########################################################################################
#########################################################################################
#########################################################################################
if phase == 'train':
    
    model.train()
    for epoch in tqdm(range(args.epochs)):
    
        step = 0
        loss_list = []
        
        for encoder_inputs, labels in train_dl:
            
            Mat_labels = label_interpolation(labels)
            Mat_encoder = encoder_interpolation(encoder_inputs, 2, 
                                                   visualize = False, Laplacian=True)

            if torch.cuda.is_available() == True:
                #interp_inputs = encoder_inputs.to(device)
                #interp_labels = labels.to(device)
                interp_inputs = Mat_encoder.to(device)
                interp_labels = Mat_labels.to(device)
                
            
                
            y_hat = model(interp_inputs, edge_index) 

            loss = loss_fn(y_hat, interp_labels) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step= step+ 1
            loss_list.append(loss.item())
            
            if step % 5 == 0:
                print(f'Epoch: {epoch}, Iteration {step}... Loss = {loss.item()}')
                print(sum(loss_list)/len(loss_list))

                print(np.max(interp_labels[0,:,0].detach().cpu().numpy()))
                error = abs(y_hat[0,:,0].detach().cpu().numpy() - 
                            interp_labels[0,:,0].detach().cpu().numpy())
                print(np.max(error), np.average(error), np.std(error))
            #torch.save(model.state_dict(), 
            #'%s/trained_cPhiGNN_LinearInterp_Laplacian.pth' % 
            #           ('save_dir'))
            #sys.exit()
