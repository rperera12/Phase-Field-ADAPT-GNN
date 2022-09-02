# Phase-Field-ADAPT-GNN
ADAPTive mesh-based Graph Neural Network (ADAPT-GNN) for simulating phase field fracture models

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

The second order phase field fracture model codes used to generate the training-, validation-, and test-set are included in
      
## Models:

### Message-Passing Model
NOTE: Found in src/models.py 

The message-passing model used was Graph Isomorphism Network with Edge Features (GINE) - from the opem-source Pytorch Geometric Libraries.
In src/models.py you will find models for various number of message-passing steps.
In the article we cover the optimal number of message-passing steps found for each model (XDisp-GNN, YDisp-GNN, and cPhi-GNN). 
But you are free to modify them and test different models.
The output from the GINE models are the vertex and edge features in the latent space for each respective model.

### Attention Temporal Graph Convolutional Networks (ATGCN)
NOTE: Found in src/models.py

The ATGCN was used to use the latent space information generated from GINE in order to propagate the system in time.
Two similar ATGCNs were used for XDisp-GNN and YDisp-GNN, and a modified ATGCN was used for cPhi-GNN.

## Dataset:

