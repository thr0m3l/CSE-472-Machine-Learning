import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


X = np.loadtxt('data2D_B1.txt')


def plot_gaussian(data, means, covariances, K, responsibilities):

    plt.scatter(data[:, 0], data[:, 1], c=responsibilities.argmax(axis=1), cmap='viridis', s=40, edgecolor='k',
                alpha=0.2, marker='.')
    x, y = np.mgrid[np.min(data[:, 0]):np.max(
        data[:, 0]):.01, np.min(data[:, 1]):np.max(data[:, 1]):.01]
    positions = np.dstack((x, y))
    for j in range(K):
        rv = multivariate_normal(means[j], covariances[j])
        plt.contour(x, y, rv.pdf(positions),
                    colors='black', alpha=0.6, linewidths=1)


def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)


def initialize_clusters(X, n_clusters):
    clusters = []
    idx = np.arange(X.shape[0])

    # Randomly generate the initial means
    mu_k = X[np.random.choice(idx, n_clusters, replace=False), :]

    for i in range(n_clusters):
        clusters.append({
            'pi_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })
    return clusters


def expectation_step(X, clusters):
    global gamma_nk, totals
    N = X.shape[0]
    K = len(clusters)
    totals = np.zeros((N, 1), dtype=np.float64)
    gamma_nk = np.zeros((N, K), dtype=np.float64)

    for k, cluster in enumerate(clusters):
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']

        gamma_nk[:, k] = (pi_k * gaussian(X, mu_k, cov_k)).ravel()

    totals = np.sum(gamma_nk, 1)
    gamma_nk /= np.expand_dims(totals, 1)


def maximization_step(X, clusters):
    global gamma_nk
    N = float(X.shape[0])

    for k, cluster in enumerate(clusters):
        gamma_k = np.expand_dims(gamma_nk[:, k], 1)
        N_k = np.sum(gamma_k, axis=0)

        pi_k = N_k / N
        mu_k = np.sum(gamma_k * X, axis=0) / N_k
        cov_k = (gamma_k * (X - mu_k)).T @ (X - mu_k) / N_k

        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k


def get_likelihood(X, clusters):
    global gamma_nk, totals
    sample_likelihoods = np.log(totals)
    return np.sum(sample_likelihoods), sample_likelihoods


def train_gmm(X, n_clusters, n_epochs):
    clusters = initialize_clusters(X, n_clusters)
    likelihoods = np.zeros((n_epochs, ))
    scores = np.zeros((X.shape[0], n_clusters))
    history = []
    dim = X.shape[1]

    if dim <= 2:
        fig = plt.figure()
        plt.ion()

    for i in range(n_epochs):
        clusters_snapshot = []

        # This is just for our later use in the graphs
        for cluster in clusters:
            clusters_snapshot.append({
                'mu_k': cluster['mu_k'].copy(),
                'cov_k': cluster['cov_k'].copy()
            })

        history.append(clusters_snapshot)

        expectation_step(X, clusters)
        maximization_step(X, clusters)

        # Extract all the means and covariances
        means = np.array([cluster['mu_k'] for cluster in clusters])
        covariances = np.array([cluster['cov_k'] for cluster in clusters])
        plt.clf()
        if dim <= 2:
            plot_gaussian(X, means, covariances, n_clusters, gamma_nk)
            plt.title("Iteration {}".format(i))
            plt.pause(0.005)

        likelihood, sample_likelihoods = get_likelihood(X, clusters)
        likelihoods[i] = likelihood

        print('Epoch: ', i + 1, 'Likelihood: ', likelihood)

    scores = np.log(gamma_nk)

    plt.ioff()
    return clusters, likelihoods, scores, sample_likelihoods, history


def plot():
    plt.figure(figsize=(10, 10))
    plt.title('Log-Likelihood')
    plt.plot(np.arange(1, n_epochs + 1), likelihoods)
    plt.show()


n_clusters = 5
n_epochs = 25

clusters, likelihoods, scores, sample_likelihoods, history = train_gmm(
    X, n_clusters, n_epochs)
