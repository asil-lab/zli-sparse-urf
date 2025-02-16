"""
This is the demo for the construction of sparse universally rigid framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import configurations as cfg

from graph import Graph
from stress import Stress
from utils import incidence_matrix_fully_connected, visualize_graph

NUM_NODES = 10
DIM = 2

# nominal_config = cfg.circular_config(NUM_NODES, radius = 5)
# incidence_full = incidence_matrix_fully_connected(NUM_NODES)

# hyperparam_dict = {'alpha':10, 'beta': 1, 'lambda': 0.1}
# stress_optimizer = Stress(nominal_config)
# stress, sparse_weights = stress_optimizer.sparse_stress(incidence_full, hyperparam_dict, sparse_tol=5e-5)

# result_incidence = np.delete(incidence_full, np.where(np.abs(sparse_weights) < 5e-5), axis=1)
# result_graph = Graph(result_incidence)
# result_graph.visualize(nominal_config)
# # print(stress)

# print(np.linalg.eigvalsh(stress))
# # visualize the nominal configuration
# plt.figure()
# plt.scatter(range(NUM_NODES), np.linalg.eigvalsh(stress), c='r')
# plt.show()

# work in yang2019
nominal_config = np.array([
    [-np.sqrt(3)/2, np.sqrt(3)/2, 0, -1/2, 1, -2/3],
    [-1/2, -1/2, 1, -1/2, 0, 8/3 - np.sqrt(3)],
    [0, 0, 0, 3, 3, 3*np.sqrt(3) - 2]
])
stress_optimizer = Stress(nominal_config)
stress = stress_optimizer.yang2019()
result_graph = Graph(stress)
visualize_graph(result_graph, nominal_config)
print(np.linalg.eigvalsh(stress))

