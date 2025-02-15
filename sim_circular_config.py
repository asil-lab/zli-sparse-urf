"""
 Simulation of the 2D circular configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
import configurations as cfg
from utils import incidence_matrix_fully_connected, affine_formation_control, visualize_graph
from stress import Stress
from graph import Graph

def my_sparse_urf(nominal_config, hyperparam_dict):

    num_nodes = nominal_config.shape[1]
    incidence_full = incidence_matrix_fully_connected(num_nodes)

    stress_optimizer = Stress(nominal_config)
    stress, sparse_weights = stress_optimizer.sparse_stress(incidence_full, hyperparam_dict, sparse_tol=5e-5)

    result_incidence = np.delete(incidence_full, np.where(np.abs(sparse_weights) < 5e-5), axis=1)
    result_graph = Graph(result_incidence)

    return stress, result_graph

def yang2019(nominal_config):

    stress_optimizer = Stress(nominal_config)
    stress = stress_optimizer.yang2019()

    result_graph = Graph(stress)

    return stress, result_graph










if __name__ == "__main__":

    
    NUM_NODES = 10
    nominal_config = cfg.circular_config(NUM_NODES, radius = 2)

    # alpha = 0.5
    stress_alpha_05, result_graph_alpha_05 = my_sparse_urf(nominal_config, {'alpha':0.5, 'beta': 1, 'lambda': 0.1})

    # alpha = 1.5
    stress_alpha_15, result_graph_alpha_15 = my_sparse_urf(nominal_config, {'alpha':1.5, 'beta': 1, 'lambda': 0.1})

    # alpha = 5
    stress_alpha_5, result_graph_alpha_5 = my_sparse_urf(nominal_config, {'alpha':5, 'beta': 1, 'lambda': 0.1})

    # visulize all the resulting graphs
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    visualize_graph(result_graph_alpha_05, nominal_config)
    
    plt.subplot(132)
    visualize_graph(result_graph_alpha_15, nominal_config)
    
    plt.subplot(133)
    visualize_graph(result_graph_alpha_5, nominal_config)

    plt.tight_layout() 
    plt.show()

    stress_yang2019, result_graph_yang2019 = yang2019(nominal_config)

    plt.figure()
    visualize_graph(result_graph_yang2019, nominal_config)
    plt.show()
