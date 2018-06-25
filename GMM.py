#!/usr/bin/env python
import numpy as np
from sklearn.cluster import KMeans


class GaussianMixture:
    def __init__(self, X, n_components=5):
        self.n_components = n_components
        self.n_features = X.shape[1]
        self.n_samples = np.zeros(self.n_components)

        self.coefs = np.zeros(self.n_components)
        self.means = np.zeros((self.n_components, self.n_features))
        # Full covariance
        self.covariances = np.zeros(
            (self.n_components, self.n_features, self.n_features))

        self.init_with_kmeans(X)

    def init_with_kmeans(self, X):
        label = KMeans(n_clusters=self.n_components, n_init=1).fit(X).labels_
        self.fit(X, label)

    def calc_score(self, X, ci):
        """Predict probabilities of samples belong to component ci

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        ci : int

        Returns
        -------
        score : array, shape (n_samples,)
        """
        score = np.zeros(X.shape[0])
        if self.coefs[ci] > 0:
            diff = X - self.means[ci]
            # _mult = diff[0].dot(np.linalg.inv(self.covariances[ci])).dot(diff[0].T)
            # _mult = diff[1].dot(np.linalg.inv(self.covariances[ci])).dot(diff[1].T)
            mult = np.einsum(
                'ij,ij->i', diff, np.dot(np.linalg.inv(self.covariances[ci]), diff.T).T)
            score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi) / \
                np.sqrt(np.linalg.det(self.covariances[ci]))

        return score

    def calc_prob(self, X):
        """Predict probability (weighted score) of samples belong to the GMM

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        prob : array, shape (n_samples,)
        """
        prob = [self.calc_score(X, ci) for ci in range(self.n_components)]
        return np.dot(self.coefs, prob)

    def which_component(self, X):
        """Predict samples belong to which GMM component

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        comp : array, shape (n_samples,)
        """
        prob = np.array([self.calc_score(X, ci)
                         for ci in range(self.n_components)]).T
        # print(prob)
        return np.argmax(prob, axis=1)

    def fit(self, X, labels):
        assert self.n_features == X.shape[1]

        self.n_samples[:] = 0
        self.coefs[:] = 0

        uni_labels, count = np.unique(labels, return_counts=True)
        self.n_samples[uni_labels] = count

        variance = 0.01
        for ci in uni_labels:
            n = self.n_samples[ci]

            self.coefs[ci] = n / np.sum(self.n_samples)
            self.means[ci] = np.mean(X[ci == labels], axis=0)
            self.covariances[ci] = 0 if self.n_samples[ci] <= 1 else np.cov(
                X[ci == labels].T)

            det = np.linalg.det(self.covariances[ci])
            if det <= 0:
                # Adds the white noise to avoid singular covariance matrix.
                self.covariances[ci] += np.eye(self.n_features) * variance
                det = np.linalg.det(self.covariances[ci])


if __name__ == '__main__':
    xx_list = np.array([[1.0, 3.0, 4],
                        [2.0, 3.0, 4],
                        [2.0, 3.5, 4],
                        [3, 4, 5],
                        [3.5, 4, 5],
                        [4, 5, 6],
                        [4, 5.5, 6],
                        [3, 6, 7]])
    center = np.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])

    gmm = GaussianMixture(xx_list)
    print(gmm.which_component(np.array([[1, 3, 4]])))

    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(xx_list[:,0], xx_list[:,1], xx_list[:,2])
    # plt.show()
