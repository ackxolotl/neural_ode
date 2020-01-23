import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def generate_concentric_shape(inner_range, outer_range, num_inner, num_outer):

    def sample_point(min_distance, max_distance):
        dist = min_distance + (max_distance - min_distance) * np.random.uniform() 
        direction = np.random.normal(loc=0, scale=1, size=2)
        direction = direction / LA.norm(direction)
        return dist * direction

    data_inner = np.array([sample_point(*inner_range) for _ in range(num_inner)])
    data_outer = np.array([sample_point(*outer_range) for _ in range(num_outer)])
    
    x_inner, y_inner = data_inner[:,0], data_inner[:,1]
    x_outer, y_outer = data_outer[:,0], data_outer[:,1]

    return (x_inner, y_inner), (x_outer, y_outer)


if __name__ == '__main__':
    # Call generator
    inner_ring, outer_ring = generate_concentric_shape(inner_range=(0, 4), outer_range=(10, 15), num_inner=100, num_outer=400)

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(*inner_ring, s=7)
    ax.scatter(*outer_ring, s=7)
    ax.grid()
    plt.show()

