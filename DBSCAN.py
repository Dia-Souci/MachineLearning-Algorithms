import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn import metrics
import os
import time


#DBSCAN FROM SKLEARN
def DBSCAN_sklearn(X,eps,min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    return labels,n_clusters_,n_noise_


DATA_PATH = os.path.join('.','data')
ORIGINAL_PATH = os.path.join(DATA_PATH, 'Data1Reduced3.xlsx')

global df

df = pd.read_excel(ORIGINAL_PATH)

X = df.to_numpy()

#load iris dataset
# from sklearn import datasets
# iris = datasets.load_iris()

# X = iris.data[:, :2]

def findNeighbors(idx,X,eps):
    neighbors = []
    for j in range(len(X)):
        if idx != j and np.linalg.norm(X[idx]-X[j]) < eps:
            neighbors.append(j)
    return neighbors


def setCorePoints(X,eps,min_samples):
    core_points = []
    for i in range(len(X)):
        if len(findNeighbors(i,X,eps)) >= min_samples:
            core_points.append(i)
    return core_points


def setBorderPoints(X,eps,min_samples,core_points):
    border_points = []
    for i in range(len(X)):
        if len(findNeighbors(i,X,eps)) < min_samples and i not in core_points:
            border_points.append(i)
    return border_points


def setNoisePoints(X,eps,min_samples,core_points,border_points):
    noise_points = []
    for i in range(len(X)):
        if i not in core_points and i not in border_points:
            noise_points.append(i)
    return noise_points

#DBSCAN FROM SCRATCH
def DBSCAN_scratch_exhaustive(X,eps,min_samples): 
    labels = np.zeros(len(X))
    c = 0
    corePoints = setCorePoints(X,eps,min_samples)
    borderPoints = setBorderPoints(X,eps,min_samples,corePoints)
    noisePoints = setNoisePoints(X,eps,min_samples,corePoints,borderPoints)
    for i in range(len(X)):
        print("Processing %d/%d" % (i+1, len(X))) 
        if i not in noisePoints:
            if i in corePoints:
                if labels[i] == 0:
                    c += 1
                    labels[i] = c
                for j in range(len(X)):
                    if j in findNeighbors(i,X,eps):
                        if labels[j] == 0:
                            labels[j] = c
                        for k in range(len(X)):
                            if k in findNeighbors(j,X,eps):
                                if labels[k] == 0:
                                    labels[k] = c
    return labels,c,len(noisePoints)

def DBSCAN_scratch(X,eps,min_samples): 
    labels = np.zeros(len(X))
    c = 0
    corePoints = setCorePoints(X,eps,min_samples)
    borderPoints = setBorderPoints(X,eps,min_samples,corePoints)
    noisePoints = setNoisePoints(X,eps,min_samples,corePoints,borderPoints)
    for i in corePoints :
        print("Processing corePoints %d/%d" % (corePoints.index(i)+1, len(corePoints)))
        if labels[i] == 0:
            c += 1
            labels[i] = c
        for j in findNeighbors(i,X,eps):
            if labels[j] == 0:
                labels[j] = c
            for k in findNeighbors(j,X,eps):
                if labels[k] == 0:
                    labels[k] = c
    return labels,c,len(noisePoints)

def meanDistance(x,X):
    distances = []
    for i in range(len(X)):
        distances.append(np.linalg.norm(x-X[i]))
    return np.mean(distances)

# means =[mean for mean in map(lambda x: meanDistance(x,X),X)]
# print(np.mean(means))

#plot results
def plotClusters(X,labels):
    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()


Eps = [0.5,1,1.5,2]
Min_samples = [3,4,5]
parameters = []
for x in Eps:
    for y in Min_samples:
        parameters.append((x,y))
times =[]
silhouette = []
number_clusters = []
for eps in Eps:
    for min_samples in Min_samples:
        start = time.time()
        labels , n_clusters_ , n_noise_ = DBSCAN_scratch(X,eps,min_samples)
        end = time.time()
        times.append(end-start)
        print('Estimated number of clusters: %d' % n_clusters_)
        number_clusters.append(n_clusters_)
        if(n_clusters_ > 1):
            score = silhouette_score(X, labels,metric='euclidean')
            silhouette.append(score)
            print("Silhouette Coefficient: %0.3f" % score)
        else :
            silhouette.append(0)

test = pd.DataFrame({'Parameters':parameters ,'Time':times,'Silhouette':silhouette,'Number_clusters':number_clusters})
test.to_excel('DBSCAN_tests.xlsx')