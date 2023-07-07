import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import igraph
import scipy.optimize as optimize

import sys
sys.path.insert(0, '../modules/')
import model 
import utils

import copy


N_modules = 12
Nodes_per_module = [75]*N_modules

N = sum(Nodes_per_module)
Nconn_per_node = 1

p_intra = 0.3
p_connect = 0.05

f = 0.5
xi = 1

Nrep = 500
NBatch = 10

kwargs = {'p_intra': p_intra, 'p_connect': p_connect,
           'Nconn_per_node': Nconn_per_node}

for idx_batch in range(NBatch):
    lambdaMax_all = utils.average_comm_removal(utils.generate_modular_ER, N_modules, Nodes_per_module,
                                               kwargs, f, xi, Nrep = Nrep, seeds = None)
    
    np.save(f'../data/FragmentationModular/FragmentationModularlambdaMax_{idx_batch}.npy', lambdaMax_all)