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


if __name__ == "__main__":

    
    NUM_NODES = 10
    nominal_config = cfg.circular_config(NUM_NODES, radius = 2)

    # alpha = 0.5
    stress_alpha_05, result_graph_alpha_05 = my_sparse_urf(nominal_config, {'alpha':0.5, 'beta': 1, 'lambda': 0.1})

    # alpha = 1.5
    stress_alpha_15, result_graph_alpha_15 = my_sparse_urf(nominal_config, {'alpha':1.5, 'beta': 1, 'lambda': 0.1})

    # alpha = 5
    stress_alpha_5, result_graph_alpha_5 = my_sparse_urf(nominal_config, {'alpha':5, 'beta': 1, 'lambda': 0.1})

    # yang2019
    from scipy.io import loadmat
    stress_yang2019 = loadmat('state-of-arts/stress_yang2019_circular.mat')['stress_mat_yang2019']
    result_graph_yang2019 = Graph(stress_yang2019)

    # lin2016 Grunbaum
    stress_lin2016_Grunbaum = loadmat('state-of-arts/stress_lin2016_Grunbaum_circular.mat')['stress_mat_lin2016_Grunbaum']
    incidence_Grunbaum_circular = loadmat('state-of-arts/stress_lin2016_Grunbaum_circular.mat')['B']
    result_graph_lin2016_Grunbaum = Graph(incidence_Grunbaum_circular)

    ################################################# plot 1 #################################################
    # visulize all the resulting graphs
    plt.figure(figsize=(8, 4))

    plt.subplot(231)
    visualize_graph(Graph(np.eye(NUM_NODES)), nominal_config)
    plt.grid()

    plt.subplot(232)
    visualize_graph(result_graph_yang2019, nominal_config)

    plt.subplot(233)
    visualize_graph(result_graph_lin2016_Grunbaum, nominal_config)

    plt.subplot(234)
    visualize_graph(result_graph_alpha_05, nominal_config)
    
    plt.subplot(235)
    visualize_graph(result_graph_alpha_15, nominal_config)
    
    plt.subplot(236)
    visualize_graph(result_graph_alpha_5, nominal_config)

    plt.tight_layout() 
    plt.savefig("figures/sparse-graphs.png", dpi=300, bbox_inches='tight')
    plt.show()


    ################################################# plot 2 #################################################
    # compare the eigenvalues of the stress matrices
    # first normalize the stress matrices
    stress_alpha_05 = stress_alpha_05 / np.linalg.norm(stress_alpha_05,2)
    stress_alpha_15 = stress_alpha_15 / np.linalg.norm(stress_alpha_15,2)
    stress_alpha_5 = stress_alpha_5 / np.linalg.norm(stress_alpha_5,2)
    stress_yang2019 = stress_yang2019 / np.linalg.norm(stress_yang2019,2)
    stress_lin2016_Grunbaum = stress_lin2016_Grunbaum / np.linalg.norm(stress_lin2016_Grunbaum,2)

    sorted_eigs_alpha_05 = np.sort(np.linalg.eigvalsh(stress_alpha_05))
    sorted_eigs_alpha_15 = np.sort(np.linalg.eigvalsh(stress_alpha_15))
    sorted_eigs_alpha_5 = np.sort(np.linalg.eigvalsh(stress_alpha_5))
    sorted_eigs_yang2019 = np.sort(np.linalg.eigvalsh(stress_yang2019))
    sorted_eigs_lin2016_Grunbaum = np.sort(np.linalg.eigvalsh(stress_lin2016_Grunbaum))

    print(f"condition number alpha 0.5: {1/sorted_eigs_alpha_05[3]}")
    print(f"condition number alpha 1.5: {1/sorted_eigs_alpha_15[3]}")
    print(f"condition number alpha 5: {1/sorted_eigs_alpha_5[3]}")
    print(f"condition number yang2019: {1/sorted_eigs_yang2019[3]}")
    print(f"condition number lin2016_Grunbaum: {1/sorted_eigs_lin2016_Grunbaum[3]}")

    x_axis = range(1,NUM_NODES+1)
    plt.figure(figsize=(4, 6))
    plt.plot(x_axis, sorted_eigs_alpha_05[::-1], label='alpha = 0.5', color='r', linestyle='-', marker='s')
    plt.plot(x_axis, sorted_eigs_alpha_15[::-1], label='alpha = 1.5', color='g', linestyle='-', marker='o')
    plt.plot(x_axis, sorted_eigs_alpha_5[::-1], label='alpha = 5', color='b', linestyle='-', marker='^')
    plt.plot(x_axis, sorted_eigs_yang2019[::-1], label='yang2019', color='m', linestyle='--', marker='x')
    plt.plot(x_axis, sorted_eigs_lin2016_Grunbaum[::-1], label='lin2016_Grunbaum', color='c', linestyle='--', marker='d')
    plt.xlim([0, NUM_NODES+1])
    plt.xticks(x_axis)
    # plt.legend()
    # plt.grid()
    plt.savefig("figures/eigenvalues.png", dpi=300, bbox_inches='tight')
    plt.show()


    ################################################# plot 3 #################################################
    # simulate the affine formation control and plot the tracking error
    max_itr = 10000
    dt = 0.01
    t = np.linspace(dt, max_itr*dt, max_itr)

    tracking_error_alpha_05 = affine_formation_control(nominal_config, stress_alpha_05, gain=50, max_itr=max_itr, dt=dt)
    tracking_error_alpha_15 = affine_formation_control(nominal_config, stress_alpha_15, gain=50, max_itr=max_itr, dt=dt)
    tracking_error_alpha_5 = affine_formation_control(nominal_config, stress_alpha_5, gain=50, max_itr=max_itr, dt=dt)
    tracking_error_yang2019 = affine_formation_control(nominal_config, stress_yang2019, gain=50, max_itr=max_itr, dt=dt)
    tracking_error_lin2016_Grunbaum = affine_formation_control(nominal_config, stress_lin2016_Grunbaum, gain=50, max_itr=max_itr, dt=dt)


    plt.figure(figsize=(4, 6))
    plt.plot(t, tracking_error_alpha_05, label='alpha = 0.5', color='r', linestyle='-')
    plt.plot(t, tracking_error_alpha_15, label='alpha = 1.5', color='g', linestyle='-')
    plt.plot(t, tracking_error_alpha_5, label='alpha = 5', color='b', linestyle='-')
    plt.plot(t, tracking_error_yang2019, label='yang2019', color='m', linestyle='--')
    plt.plot(t, tracking_error_lin2016_Grunbaum, label='lin2016_Grunbaum', color='c', linestyle='--')
    # plt.legend()

    plt.yscale('log')
    plt.savefig("figures/tracking-error.png", dpi=300, bbox_inches='tight')
    plt.show()


