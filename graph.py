"""
    The class for graphs
"""

import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, matrix):
        """
        num_nodes: int, number of nodes
        matrix: numpy array, graph representation matrix, could be adjacency, laplacian, or incidence
        """

        self.num_nodes = matrix.shape[0]
        matrix_type = self.detect_matrix_type(matrix)
        self.edges = self.matrix_to_edges(matrix, matrix_type)

        
    def detect_matrix_type(self, matrix):

        """Detect whether the given matrix is adjacency, laplacian, or incidence."""
        
        n, m = matrix.shape
        
        if n == m:  # square matrix
            if np.allclose(matrix, matrix.T):  # symmetric
                if np.linalg.norm(np.diag(matrix),np.inf) > 1e-6:  # row sums equal diagonal
                    print('Laplacian detected')
                    return 'laplacian'
                else:
                    print('Adjacency detected')
                    return 'adjacency'
        else:
            print('Incidence detected')
            return 'incidence'
    
    def matrix_to_edges(self, matrix, matrix_type):
        """
        Interprets different kinds of matrices (Laplacian, incidence, adjacency) to a set of edges.
        
        Parameters:
            matrix (np.ndarray): Input matrix.
            matrix_type (str): Type of the matrix ('laplacian', 'incidence', 'adjacency').
        
        Returns:
            list of tuples: List of edges represented as (node1, node2).
        """
        edges = []
        n = matrix.shape[0]
        
        if matrix_type == 'adjacency':
            for i in range(n):
                for j in range(i + 1, n):
                    if matrix[i, j] != 0:
                        edges.append((i, j))
        
        elif matrix_type == 'laplacian':
            for i in range(n):
                for j in range(i + 1, n):
                    if np.abs(matrix[i, j]) > 1e-6:  
                        edges.append((i, j))
        
        elif matrix_type == 'incidence':
            _, m = matrix.shape
            for edge_index in range(m):
                nodes = np.where(matrix[:, edge_index] != 0)[0]
                if len(nodes) == 2:
                    edges.append((nodes[0], nodes[1]))
        
        return edges
    
    def adjacency_mat(self):
        """
        Converts a list of edges into an adjacency matrix.
        
        Returns:
            np.ndarray: Adjacency matrix of shape (n, n).
        """
        adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        
        for i, j in self.edges:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1  # Since the graph is undirected
        
        return adjacency_matrix
    
    def laplacian_mat(self):
        """
        Converts a list of edges into a Laplacian matrix.
        
        Returns:
            np.ndarray: Laplacian matrix of shape (n, n).
        """
        adjacency_matrix = self.adjacency_mat()
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        
        return degree_matrix - adjacency_matrix
    
    def incidence_mat(self):
        """
        Converts a list of edges into an incidence matrix.
        
        Returns:
            np.ndarray: Incidence matrix of shape (n, m).
        """
        incidence_matrix = np.zeros((self.num_nodes, len(self.edges)), dtype=int)
        
        for i, (node1, node2) in enumerate(self.edges):
            incidence_matrix[node1, i] = 1
            incidence_matrix[node2, i] = -1
        
        return incidence_matrix
    
    def degree_mat(self):
        """
        Returns the degree matrix of the graph.
        
        Returns:
            np.ndarray: Degree matrix of shape (n, n).
        """
        return np.diag(np.sum(self.adjacency_mat(), axis=1))
    
    
    
    
    
    