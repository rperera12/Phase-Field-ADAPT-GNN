# Phase-Field-ADAPT-GNN
ADAPTive mesh-based Graph Neural Network (ADAPT-GNN) for simulating phase field fracture models

![Sample Animation](Supplementary_Figure_3.gif "Phase-field VS. ADAPT-GNN")

Should you find this repository as a useful tool for research or application, please kindly cite the original article [Dynamic and adaptive mesh-based graph neural network framework for simulating displacement and crack fields in phase field models](https://arxiv.org/abs/2208.14364v2)

      @misc{https://doi.org/10.48550/arxiv.2208.14364,
        doi = {10.48550/ARXIV.2208.14364},
        url = {https://arxiv.org/abs/2208.14364},
        author = {Perera, Roberto and Agrawal, Vinamra},
        keywords = {Materials Science (cond-mat.mtrl-sci), FOS: Physical sciences, FOS: Physical sciences},
        title = {Dynamic and adaptive mesh-based graph neural network framework for simulating displacement and crack fields in phase field models},
        publisher = {arXiv},
        year = {2022},
        copyright = {arXiv.org perpetual, non-exclusive license}
      }

## Adaptive mesh refinement (AMR) & Phase field (PF) fracture model

The second order phase field fracture model codes used to generate the training-, validation-, and test-set are included in directory: Phase-Field-Fracture-Model
The adaptive mesh refinement used in the developed ADAPT-GNN framework is also included in directory: Phase-Field-Fracture-Model

We thank the authors in "Adaptive fourth-order phase field analysis for brittle fracture" for providing the open-source phase field model found in [IGAPack-PhaseField](https://github.com/somdattagoswami/IGAPack-PhaseField)

      @article{GOSWAMI2020112808,
            title = {Adaptive fourth-order phase field analysis for brittle fracture},
            journal = {Computer Methods in Applied Mechanics and Engineering},
            volume = {361},
            pages = {112808},
            year = {2020},
            issn = {0045-7825},
            doi = {https://doi.org/10.1016/j.cma.2019.112808},
            url = {https://www.sciencedirect.com/science/article/pii/S0045782519307005},
            author = {Somdatta Goswami and Cosmin Anitescu and Timon Rabczuk},
            keywords = {Brittle fracture, Fourth-order, Phase field, Adaptive, Stress-degradation, PHT-splines}
      }

### Changing initial crack length, angle, and edge position

#### Crack edge position and crack angle in code: Phase-Field-Fracture-Model/SingleEdgeTension.m

1) Line 14, adder = 0.05+0.05*i, defines the edge position in a for loop iterating through "i". 
   E.g., when i=5, the crack edge position will be at the left edge with y = 0.30cm
    
2) To change the crack angle, line 15, angles_stored = [45], defines a list with the angle in degrees.
   Note that you can include more angles in the list angles_stored to iterate through a various crack angles
   
#### Crack length in code: Phase-Field-Fracture-Model/utils/history_tensile.m

2) Line 13, crack_size = 0.1, defines the value of the desired crack length.
   Note that you can include a global variable in code: <Phase-Field-Fracture-Model/SingleEdgeTension.m> to iterate through a series of crack lengths


## Training files:
NOTE: Found in src/train_cPhi_LinearInterp_Laplacian.py 

New updated trainer file for cPhi_GNN using Attention Temporal Graph Convolutional Networks (ATGCN).
Laplacian node features included.
Training data sample included in src/train_pt_cPhi/16_Case.
Pretrained weights can be found in src/save_dir/


## Models:

### Message-Passing Model
NOTE: Found in src/models.py 

The message-passing model used was Graph Isomorphism Network with Edge Features (GINE) - from the open-source Pytorch Geometric Libraries.
In src/models.py you will find models for various number of message-passing steps.
In the article we cover the optimal number of message-passing steps found for each model (XDisp-GNN, YDisp-GNN, and cPhi-GNN). 
But you are free to modify them and test different models.
The output from the GINE models are the vertex and edge features in the latent space for each respective model.

### Attention Temporal Graph Convolutional Networks (ATGCN)
NOTE: Found in src/models.py

The ATGCN uses the latent space information generated from GINE in order to propagate the system in time.
Two similar ATGCNs were used for XDisp-GNN and YDisp-GNN, and a modified ATGCN was used for cPhi-GNN.

## Utilities:
NOTE: Found in src/utils.py 

All utilities/functions including the nearest neighbors generation (nodes and edges), the adaptive mesh refinement generation, nodes and edges features generators, a function for converting the phase field fracture model's .mat files to numpy and .pt format, and more can be found here.  


## Dataloader:
NOTE: Found in src/dataloader.py 

Includes the dataloader class for XDisp-, YDisp-, and cPhi-GNN. The format is developed for compatibility with Pytorch Geometric (PyG)   


## Arguments:
NOTE: Found in src/config_HPC_Latest.py

Arguments used are described


