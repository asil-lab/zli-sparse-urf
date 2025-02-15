import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

# Step 1: Define a given configuration in 3D
def generate_configuration():
    return np.array([
        [-np.sqrt(3)/2, np.sqrt(3)/2, 0, -1/2, 1, -2/3],
        [-1/2, -1/2, 1, -1/2, 0, 8/3],
        [0, 0, 0, 3, 3, 3]
    ])

# Step 2: Compute extended configuration matrix

def compute_extended_matrix(q):
    n = q.shape[1]
    ones_row = np.ones((1, n))
    return np.vstack((q, ones_row))

# Step 3: Compute a sparse Gale matrix

def compute_gale_matrix(Q):
    return null_space(Q.T)

# Step 4: Compute stress matrix (Ω)
def compute_stress_matrix(D):
    return D @ D.T

# Step 5: Assign cables and struts
def classify_edges(Omega):
    edges = []
    for i in range(Omega.shape[0]):
        for j in range(i + 1, Omega.shape[1]):
            if Omega[i, j] < 0:
                edges.append((i, j, 'cable'))
            elif Omega[i, j] > 0:
                edges.append((i, j, 'strut'))
    return edges

# Step 6: Simulate formation control
def simulate_formation_control(q, edges, steps=100, dt=0.01):
    q_traj = [q.copy()]
    velocities = np.zeros_like(q)
    for _ in range(steps):
        forces = np.zeros_like(q)
        for i, j, type_ in edges:
            diff = q[:, j] - q[:, i]
            dist = np.linalg.norm(diff)
            force = 0.1 * diff / (dist + 1e-6)  # Small value to prevent div by zero
            if type_ == 'cable':
                forces[:, i] += force
                forces[:, j] -= force
            elif type_ == 'strut':
                forces[:, i] -= force
                forces[:, j] += force
        velocities += forces * dt
        q += velocities * dt
        q_traj.append(q.copy())
    return np.array(q_traj)

# Main execution
q = generate_configuration()
Q_star = compute_extended_matrix(q)
D = compute_gale_matrix(Q_star)
Omega = compute_stress_matrix(D)
edges = classify_edges(Omega)
trajectory = simulate_formation_control(q, edges)

# Plot final formation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(q[0, :], q[1, :], q[2, :], color='blue', label='Agents')
for i, j, type_ in edges:
    ax.plot([q[0, i], q[0, j]], [q[1, i], q[1, j]], [q[2, i], q[2, j]], color='r' if type_=='cable' else 'g')
ax.set_title("Final Tensegrity Formation")
plt.legend()
plt.show()

print(Omega)