import numpy as np

def circular_config(num_nodes, radius = 1):
    """
    Generate positions of nodes placed evenly on a circle (regular polygon).
    
    Parameters:
        r (float): Radius of the circle.
        num_nodes (int): Number of nocdes to be placed.
    
    Returns:
        np.ndarray: A 2xN array where the first row is x-coordinates and the second row is y-coordinates.
    """
    # Create an array of angles excluding 2π
    angles = np.linspace(0, 2 * np.pi, num_nodes + 1)[:-1]
    
    # Compute x and y coordinates
    x_positions = radius * np.cos(angles)
    y_positions = radius * np.sin(angles)
    
    # Stack into a 2xN array
    positions = np.vstack((x_positions, y_positions))
    
    return positions