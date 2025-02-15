import numpy as np
import matplotlib.pyplot as plt


def incidence_matrix_fully_connected(n):
        """
        Generates the incidence matrix for a fully connected graph with n nodes.
        
        Returns:
            np.ndarray: Incidence matrix of shape (num_nodes, num_edges)
        """
        # Calculate the number of edges
        num_edges = n * (n- 1) // 2
        
        # Initialize the incidence matrix with zeros
        inc_mat = np.zeros((n, num_edges), dtype=int)
        
        # Initialize edge counter
        edge_counter = 0
        
        # Fill in the incidence matrix
        for i in range(n - 1):
            for j in range(i + 1, n):
                inc_mat[i, edge_counter] = 1
                inc_mat[j, edge_counter] = -1
                edge_counter += 1
        
        return inc_mat

def affine_formation_control(target_config, stress_mat, gain = 1, max_itr = 10000, dt = 0.01):

    D, N = target_config.shape
    config = np.random.rand(D, N)
    tracking_error = np.zeros(max_itr)
    
    for i in range(max_itr):
        control_input = stress_mat @ config.T
        config -= 10 * dt * control_input.T
        config[:, :D+1] = target_config[:, :D+1]
        tracking_error[i] = np.linalg.norm(config - target_config)
    
    return tracking_error
     

def visualize_graph(graph, node_config):
    """
    Visualizes the graph in 2D or 3D.
    
    Parameters:
        graph: A graph object with edges.
        node_config (numpy.ndarray): A dxn matrix containing node coordinates.
    """
    d, n = node_config.shape
    
    if d == 2:
        # plt.figure()
        for i, j in graph.edges:
            plt.plot([node_config[0, i], node_config[0, j]], 
                     [node_config[1, i], node_config[1, j]], 'k-', zorder=1)
        
        plt.scatter(node_config[0, :], node_config[1, :], c='r', s=100, zorder=2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        # plt.show()
    
    elif d == 3:
        from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i, j in graph.edges:
            ax.plot([node_config[0, i], node_config[0, j]], 
                    [node_config[1, i], node_config[1, j]], 
                    [node_config[2, i], node_config[2, j]], 'k-')
        
        ax.scatter(node_config[0, :], node_config[1, :], node_config[2, :], c='r', s=100)
        ax.set_box_aspect([1,1,1])  # Equal aspect ratio
        # plt.axis('off')
        # plt.show()
     
     