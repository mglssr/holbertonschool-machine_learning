#!/usr/bin/env python3
"""task 11"""
import sklearn.mixture


def gmm(X, k):
    """function that calculates a GMM from a dataset"""
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    labels = gmm.predict(X)
    return gmm.weights_, gmm.means_, gmm.covariances_, labels, gmm.bic(X)
