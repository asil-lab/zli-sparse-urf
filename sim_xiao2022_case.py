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
if __name__ == "__main__":

    NUM_NODES = 10
    nominal_config = np.array([[2,1,1.5,0,0.5,-1],[0,1,-1,1.5,-1,0.5]])


    # alpha = 0.8
    stress_alpha_08, result_graph_alpha_08 = my_sparse_urf(nominal_config, {'alpha':0.8, 'beta': 1, 'lambda': 0.1})

    # yang2019
    from scipy.io import loadmat
    stress_yang2019 = loadmat('state-of-arts/stress_yang2019_xiao.mat')['stress_mat_yang2019']
    result_graph_yang2019 = Graph(stress_yang2019)


    # xiao2022
    stress_xiao2022 = np.array([[0.69,-0.55,-0.32,0,-0.09,0.28],[-0.55,0.87,0.16,-0.47,0,0],[-0.32,0.16,0.4,0,-0.24,0],
                                [0,-0.47,0,0.59,0.24,-0.36],[-0.09,0,-0.24,0.24,0.43,-0.33],[0.28,0,0,-0.36,-0.33,0.4]])
    result_graph_xiao2022= Graph(stress_xiao2022)

    # # lin2016 Cauchy
    stress_lin2016_Cauchy = loadmat('state-of-arts/stress_lin2016_Cauchy_xiao.mat')['stress_mat_lin2016_cauchy']
    incidence_Cauchy_circular = loadmat('state-of-arts/stress_lin2016_Cauchy_xiao.mat')['B']
    result_graph_lin2016_Cauchy = Graph(incidence_Cauchy_circular)

    ################################################# plot 1 #################################################
    # visulize all the resulting graphs
    plt.figure(figsize=(8, 4))


    plt.subplot(141)
    visualize_graph(result_graph_yang2019, nominal_config)

    plt.subplot(142)
    visualize_graph(result_graph_xiao2022, nominal_config)

    plt.subplot(143)
    visualize_graph(result_graph_lin2016_Cauchy, nominal_config)

    plt.subplot(144)
    visualize_graph(result_graph_alpha_08, nominal_config)


    plt.tight_layout() 
    # plt.savefig("figures/sparse-graphs-circular.png", dpi=300, bbox_inches='tight')
    plt.show()