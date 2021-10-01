import nnmt
import numpy as np
import matplotlib.pyplot as plt


def normalized_covariance_width(network):
    N = network.network_params['N']

    p = network.network_params['p']
    w = network.network_params['w']
    r = network.network_params['r']

    A = (np.eye(N) - p*w)
    B = np.linalg.inv(A) * np.linalg.inv(A.T)

    C = np.sqrt(((1 / (1 - r**2))**2 - 1) / N)
    return C / B.diagonal()[0]


network = nnmt.models.HomogeneousBinomial('homogeneous_binomial.yaml')

steps = 100
deltas = np.zeros(steps)
rs = np.linspace(0.01, 0.99, steps)
for i, r in enumerate(rs):
    network = network.change_parameters({'r': r})
    deltas[i] = normalized_covariance_width(network)

plt.semilogy(rs, deltas)
plt.xlim([-0.1, 1.1])
plt.show()