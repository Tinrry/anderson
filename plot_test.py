import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss, valitate_loss, test_loss: int):
    steps = np.array(range(len(train_loss)))
    plt.plot(steps+1, train_loss, '-o', label='train loss')
    plt.plot(steps+1, valitate_loss, '-o', label='validate loss')
    plt.plot(steps[-1]+1, test_loss, '1', label='test loss')
    plt.legend()
    plt.show()


plot_loss([1, 2, 3], [2, 1, 3], 0.1)