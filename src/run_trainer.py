import os

train_type = 'optimal'
models = ['cPhi_GNN'] #,'cPhi_GNN']
cross_valid = train_type

if train_type == 'optimal':
    for model in models:
        if model == 'cPhi_GNN':
            lr = 0.0005
            num_conv = 4
            Loss_Fn = 'SmoothL1Loss'
            vert_ed_filter = 0
            Interp_Opt = 'Uninterpolate'
            Network = 'Unormalized'

            os.system("python Test_Temporal.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {} --Train_Type {} --Network_Type {}".format(model, cross_valid, Loss_Fn, lr, int(vert_ed_filter), int(num_conv), train_type, Network))
        
        elif model == 'XDisp_GNN':
            lr = 0.001
            num_conv = 1
            Loss_Fn = 'L1Loss'
            vert_ed_filter = 4
            Interp_Opt = 'Uninterpolate'
            Network = 'Unormalized'
            
            os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {} --Train_Type {} --Network_Type {}".format(model, cross_valid, Loss_Fn, lr, int(vert_ed_filter), int(num_conv), train_type, Network))

        elif model == 'YDisp_GNN':
            lr = 0.0005
            num_conv = 4
            Loss_Fn = 'MSE'
            vert_ed_filter = 5
            Interp_Opt = 'Uninterpolate'
            Network = 'Unormalized'
            #Network = 'Normalized'
            
            os.system("python YDisp_Temporal.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {} --Train_Type {} --Network_Type {}".format(model, cross_valid, Loss_Fn, lr, int(vert_ed_filter), int(num_conv), train_type, Network))

        elif model == 'SVM_GNN':
            lr = 0.0005
            num_conv = 1
            Loss_Fn = 'MSE'
            vert_ed_filter = 1
            Interp_Opt = 'Uninterpolate'
            Network = 'Unormalized'

            os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {} --Train_Type {} --Network_Type {}".format(model, cross_valid, Loss_Fn, lr, int(vert_ed_filter), int(num_conv), train_type, Network))

elif train_type != 'optimal':
    
    _a1 = ['YDisp_GNN']
    _CV = ['LR', 'Loss_Funct'] #,'vertex_edge_filter', 'N_Conv', 'Interpolation', 'network_type'] #, 'network_type']
    
    for a1 in _a1:

        for cross_valid in _CV:

            print(' (i) Current GNN model is: %s  \n (ii) Current Cross-valid type is: %s' % (a1,cross_valid))

            if cross_valid == 'vertex_edge_filter':
                lr = 0.001
                num_conv = 3
                Loss_Fn = 'MSE'
                vert_ed_filter = [2,3,4,5]
                
                for j in range(len(vert_ed_filter)):
                    print('\n ############################# \n ----------------------------- \n Current vertex_edge_filter is: %a \n ----------------------------- \n ############################# \n' % vert_ed_filter[j])
                    os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {}".format(a1, cross_valid, Loss_Fn, lr, int(vert_ed_filter[j]), int(num_conv)))
                        
            elif cross_valid == 'N_Conv':
                
                lr = 0.001
                num_conv = [1,2,3,4,5,6]
                Loss_Fn = 'MSE'
                vert_ed_filter = 2
                
                for j in range(len(num_conv)):
                    print('\n ############################# \n ----------------------------- \n Current convolutions number is: %a \n ----------------------------- \n ############################# \n' % num_conv[j])
                    print('Current Number of Convolutions is  ', num_conv[j])
                    os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {}".format(a1, cross_valid, Loss_Fn, lr, int(vert_ed_filter), int(num_conv[j])))
                        


            elif cross_valid == 'LR':
                
                lr = [0.001, 0.005, 0.01, 0.05]
                num_conv = 3
                Loss_Fn = 'MSE'
                vert_ed_filter = 2
                
                for j in range(len(lr)):
                    print('Current Learning Rate is  ', lr[j])
                    os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {}".format(a1, cross_valid, Loss_Fn, lr[j], int(vert_ed_filter), int(num_conv)))
                        


            elif cross_valid == 'Interpolation':
                
                lr = 0.001
                num_conv = 3
                Loss_Fn = 'MSE'
                vert_ed_filter = 2
                Interp_Opt = ['Interpolate', 'Uninterpolate']
                
                for j in range(len(Interp_Opt)):
                    print('Current Number of Convolutions is  ', Interp_Opt[j])
                    os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {} --Interpolate_Option {}".format(a1, cross_valid, Loss_Fn, lr, int(vert_ed_filter), int(num_conv), Interp_Opt[j] )) 


            elif cross_valid == 'Loss_Funct':
                
                lr = 0.001
                num_conv = 3
                Loss_Fn = ['SmoothL1Loss', 'L1Loss', 'MSE'] 
                vert_ed_filter = 2
                Interp_Opt = 'Uninterpolate'
                
                for j in range(len(Loss_Fn)):
                    print('Current Loss Function is  ', Loss_Fn[j])
                    os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {} --Interpolate_Option {}".format(a1, cross_valid, Loss_Fn[j], lr, int(vert_ed_filter), int(num_conv), Interp_Opt )) 


            elif cross_valid == 'network_type':
                
                lr = 0.001
                num_conv = 3
                Loss_Fn = 'MSE'
                vert_ed_filter = 2
                Interp_Opt = 'Uninterpolate'
                Network = ['Normalized', 'Unormalized']
                
                for j in range(len(Network)):
                    print('Current Network Type is  ', Network[j])
                    os.system("python trainer_HPC_Latest.py --GNN_Model {} --cross_valid_type {} --Loss_Fn {} --learning_rate {} --vertex_edge_filter {} --nconvolutions {} --Interpolate_Option {} --Network_Type {}".format(a1, cross_valid, Loss_Fn, lr, int(vert_ed_filter), int(num_conv), Interp_Opt, Network[j] )) 