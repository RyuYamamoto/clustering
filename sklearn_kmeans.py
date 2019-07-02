# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
from sklearn import datasets 
from sklearn.cluster import KMeans

N_CLUSTERS = 5

def main():
    dataset = datasets.make_blobs(centers=N_CLUSTERS)

    features = dataset[0]

    cls = KMeans(n_clusters=N_CLUSTERS)
    pred = cls.fit_predict(features)

    for index in range(N_CLUSTERS):
        labels = features[pred == index]
        plt.scatter(labels[:, 0], labels[:, 1])

    centers = cls.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')

    plt.show()

if __name__ == '__main__':
    main()
