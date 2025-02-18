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

    NUM_NODES = 6
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
    # plt.savefig("figures/sparse-graphs-xiao.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # plot individual graph
    plt.figure(figsize=(3, 3))
    visualize_graph(result_graph_alpha_08, nominal_config)
    plt.savefig("figures/graph-alpha-08-xiao.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(3, 3))
    visualize_graph(result_graph_yang2019, nominal_config)
    plt.savefig("figures/graph-yang2019-xiao.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(3, 3))
    visualize_graph(result_graph_xiao2022, nominal_config)
    plt.savefig("figures/graph-xiao2022-xiao.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(3, 3))
    visualize_graph(result_graph_lin2016_Cauchy, nominal_config)
    plt.savefig("figures/graph-lin2016-Cauchy-xiao.png", dpi=300, bbox_inches='tight')


    ################################################# plot 2 #################################################
    # compare the eigenvalues of the stress matrices
    # first normalize the stress matrices

    stress_alpha_08 = stress_alpha_08 / np.linalg.norm(stress_alpha_08,2)
    stress_yang2019 = stress_yang2019 / np.linalg.norm(stress_yang2019,2)
    stress_xiao2022 = stress_xiao2022 / np.linalg.norm(stress_xiao2022,2)
    stress_lin2016_Cauchy = stress_lin2016_Cauchy / np.linalg.norm(stress_lin2016_Cauchy,2)

    sorted_eigs_alpha_08 = np.sort(np.linalg.eigvalsh(stress_alpha_08))
    sorted_eigs_yang2019 = np.sort(np.linalg.eigvalsh(stress_yang2019))
    sorted_eigs_xiao2022 = np.sort(np.linalg.eigvalsh(stress_xiao2022))
    sorted_eigs_lin2016_Cauchy = np.sort(np.linalg.eigvalsh(stress_lin2016_Cauchy))

    print(f"condition number alpha 0.5: {1/sorted_eigs_alpha_08[3]}")
    print(f"condition number yang2019: {1/sorted_eigs_yang2019[3]}")
    print(f"condition number xiao2022: {1/sorted_eigs_xiao2022[3]}")
    print(f"condition number lin2016_Cauchy: {1/sorted_eigs_lin2016_Cauchy[3]}")
    

    x_axis = np.arange(1, NUM_NODES+1)
    plt.figure(figsize=(4, 6))
    plt.plot(x_axis, sorted_eigs_alpha_08[::-1], label='alpha 0.8',color='r', linestyle='-', marker='s')
    plt.plot(x_axis, sorted_eigs_yang2019[::-1], label='yang2019',color='m', linestyle='--', marker='x')
    plt.plot(x_axis, sorted_eigs_xiao2022[::-1], label='xiao2022',color='g', linestyle='--', marker='o')
    plt.plot(x_axis, sorted_eigs_lin2016_Cauchy[::-1], label='lin2016 Cauchy',color='c', linestyle='--', marker='d')

    plt.xlabel('index')
    plt.ylabel('eigenvalues')

    plt.legend()
    plt.tight_layout()

    plt.savefig("figures/eigenvalues-xiao.png", dpi=300, bbox_inches='tight')
    # plt.show()

    ################################################# plot 3 #################################################

    max_itr = 10000
    dt = 0.001
    t = np.linspace(dt, max_itr*dt, max_itr)

    tracking_error_alpha_08 = affine_formation_control(nominal_config, stress_alpha_08, gain = 10, max_itr = max_itr, dt = dt)
    tracking_error_yang2019 = affine_formation_control(nominal_config, stress_yang2019, gain = 10, max_itr = max_itr, dt = dt)
    tracking_error_xiao2022 = affine_formation_control(nominal_config, stress_xiao2022, gain = 10, max_itr = max_itr, dt = dt)
    tracking_error_lin2016_Cauchy = affine_formation_control(nominal_config, stress_lin2016_Cauchy, gain = 10, max_itr = max_itr, dt = dt)

    plt.figure(figsize=(4, 6))
    plt.plot(t, tracking_error_alpha_08, label='alpha 0.8',color='r', linestyle='-')
    plt.plot(t, tracking_error_yang2019, label='yang2019',color='m', linestyle='--')
    plt.plot(t, tracking_error_xiao2022, label='xiao2022',color='g', linestyle='--')
    plt.plot(t, tracking_error_lin2016_Cauchy, label='lin2016 Cauchy',color='c', linestyle='--')

    plt.xlabel('time')
    plt.ylabel('tracking error')
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')

    plt.savefig("figures/tracking-error-xiao.png", dpi=300, bbox_inches='tight')
    plt.show()


                                         

