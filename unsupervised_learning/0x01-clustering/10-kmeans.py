#!/usr/bin/env python3
"""task 10"""
import sklearn.cluster


def kmeans(X, k):
    """function that performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
