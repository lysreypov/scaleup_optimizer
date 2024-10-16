import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(optimizer):
    min_f = np.minimum.accumulate(optimizer.Y_iters)

    plt.plot(min_f, marker='o', label='min f(x)')
    plt.xlabel('Number of steps n')
    plt.ylabel('min f(x) after n steps')
    plt.title('Plot Convergence')
    plt.grid(True)
    plt.legend()
    plt.show()