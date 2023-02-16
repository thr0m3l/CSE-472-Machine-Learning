import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

import GaussianMixture

params = dict(k=3, max_iter=100, tol=1e-6)
gmm = GaussianMixture.GaussianMixture(params)
gmm.load_data('data2D.txt')
gmm.train()

print(gmm.mu)
print(gmm.cov)
print(gmm.pi)
print(gmm.log_likelihood)
