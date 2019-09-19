import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.cluster.vq import vq, kmeans


class FixedMeanGMM:
    """ Model to estimate gaussian mixture with fixed mean. """

    def __init__(self, n_components, mean, max_iter=100, random_state=None,
                 tol=1e-10, verbose=True):
        self.n_components = n_components
        self.mean = mean
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X):
        np.random.seed(self.random_state)
        n_obs, n_features = X.shape

        # Initialise the covariance matrix using K-Means
        if self.verbose:
            print(
                'Initialising using K-Means with %i components' % self.n_components)
            start = time.time()

        centroids, _ = kmeans(X, self.n_components)
        idx, _ = vq(X, centroids)
        self.cov_ = [np.cov(X[idx == i].T) for i in range(self.n_components)]

        if self.verbose:
            print('Time taken: %.2f seconds' % (time.time() - start))

        # EM loop until convergence
        if self.verbose:
            print('Training...')
            start = time.time()

        i = 0
        loglikelihoods = []
        for i in range(self.max_iter):
            new_covs, lik = self.updated_covariances(X)
            loglikelihoods.append(lik)
            # if np.abs(post-self.post_) < self.tol:
            #     break
            # else:
            self.cov_ = new_covs
            self.lik_ = lik

            if self.verbose:
                print('Epoch %i, updated log likelihood: %.2f' % (i, lik))
        self.n_iter_ = i

        if self.verbose:
            print('Time taken: %.2f seconds' % (time.time() - start))
            print('Complete')
        return np.array(loglikelihoods)

    def updated_covariances(self, X):
        """ A single iteration """
        # E-step: estimate probability of each cluster given cluster covariances
        cluster_posterior = self.predict_proba(X)

        # M-step: update cluster covariances as weighted average of observations
        weights = (cluster_posterior.T / (
                cluster_posterior.sum(axis=1) + 1e-20)
                   ).T
        new_covs = np.matmul(weights[:, np.newaxis, :] * X.T[np.newaxis, :, :],
                             X)

        # Log likelihood
        log_lik = self.log_likelihood(X)
        return new_covs, log_lik

    def predict_proba(self, X):
        likelihood = np.stack(
            [multivariate_normal.pdf(X, mean=self.mean, cov=cov)
             for cov in self.cov_])
        cluster_posterior = (likelihood / (likelihood.sum(axis=0) + 1e-20))
        return cluster_posterior

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)

    def log_likelihood(self, X):
        return np.max(np.stack(
            [multivariate_normal.logpdf(X, mean=self.mean, cov=cov)
             for cov in self.cov_]
        ), axis=0).sum()
