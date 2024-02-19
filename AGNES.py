import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.metrics import silhouette_score
import os
import time

#Agglomerative Clustering from scratch
def agglomerative_clustering(X, n_clusters):
    n_samples = X.shape[0]
    c=0
    # Initialize clusters
    clusters = [[i] for i in range(n_samples)]
    # Compute distances between clusters
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distances[i,j] = np.linalg.norm(X[i] - X[j])
    # Loop until we have n_clusters
    while len(clusters) > n_clusters:
        # Find the two closest clusters
        min_dist = np.inf
        c1=-1
        c2=-1
        for i in range(distances.shape[0]):
            for j in range(i+1, distances.shape[0]):
                if distances[i,j] < min_dist:
                    min_dist = distances[i,j]
                    c1, c2 = i, j
        # Merge the two closest clusters
        if c1 != -1 and c2 != -1:
            clusters[c1] = clusters[c1] + clusters[c2]
        print(clusters[c1])
        clusters.pop(c2)
        # Update distances
        distances = np.delete(distances, c2, 0)
        distances = np.delete(distances, c2, 1)
        c+=1

        for i in range(distances.shape[0]):
            if i != c1:
                distances[i,c1] = np.mean([np.linalg.norm(X[i] - X[j]) for j in clusters[c1]])
                distances[c1,i] = distances[i,c1]
    return clusters

#Agglomerative Clustering using sklearn
def agglomerative_clustering_sklearn(X, n_clusters):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    clusters = []
    for i in range(n_clusters):
        clusters.append([])
    for i in range(len(X)):
        clusters[clustering.labels_[i]].append(i)
    return clusters

#Plot the clusters
def plot_clusters(X, clusters):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            plt.scatter(X[clusters[i][j],0], X[clusters[i][j],1], color=colors[i])
    plt.show()

def plotDendrogram(X):
    import scipy.cluster.hierarchy as shc
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    dend = shc.dendrogram(shc.linkage(X, method='ward'))
    plt.show()

#Sillouette score
def silhouette_score_scratch(X, clusters,a,b):
    return (b-a)/np.maximum(a,b)

def dist_IntraCluster_scratch(X, clusters):
    n_clusters = len(clusters)
    a = np.zeros(n_clusters)
    for i in range(n_clusters):
        dist = np.zeros(len(clusters[i]))
        for j in clusters[i]:
            dist[j] = np.mean([np.linalg.norm(X[j] - X[k]) for k in clusters[i]])
        a[i] = np.mean(dist)
    return np.mean(a)

def dist_InterCluster_scratch(X, clusters):
    n_samples = X.shape[0]
    n_clusters = len(clusters)
    b = np.zeros(n_clusters)
    for i in range(n_clusters):
        b[i] = np.inf
        for j in range(n_clusters):
            if i not in clusters[j]:
                dist = np.zeros(len(clusters[i]))
                for n in clusters[j]:
                    dist[n] = np.mean([np.linalg.norm(X[n] - X[k]) for k in clusters[i]])
                b[i] = min(b[i], dist)
    return np.mean(b)





#Load data from excel file data1Reduced3.xlsx
DATA_PATH = os.path.join('.','data')
ORIGINAL_PATH = os.path.join(DATA_PATH, 'Data1Reduced3.xlsx')

global df

df = pd.read_excel(ORIGINAL_PATH)
X = df.to_numpy()

#Run agglomerative clustering from scratch
# start = time.time()
# clusters = agglomerative_clustering(X, 20)
# end = time.time()


# print("Execution ended ")

# print("time taken : "+str(end-start))
# labels = np.zeros(len(X))
# for i in range(len(clusters)):
#     for j in range(len(clusters[i])):
#         labels[clusters[i][j]] = i


# print("number of clusters: ", len(clusters))

#Saving the labels to a excel file
# df['labels'] = labels
# df.to_excel('labels.xlsx')

# plotDendrogram(X)

ClusterNumber = [30,20, 15, 10, 5]
silloutte = []
Intra =[]
Inter = []
times = []
for i in ClusterNumber:
    print("Execution for cluster number %d started ",(i))

    start = time.time()
    clusters = agglomerative_clustering(X, i)
    end = time.time()
    times.append(end-start)
    print("Execution ended ")
    print("time taken : "+str(end-start))
    labels = np.zeros(len(X))
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            labels[clusters[i][j]] = i
    print("number of clusters: ", len(clusters))
    score = silhouette_score(X, labels, metric='euclidean')
    silloutte.append(score)
    print("Sillouette score: ", score)
    
    
    print("")

test = pd.DataFrame({'ClusterNumber': ClusterNumber, 'Silhouette Score': silloutte, 'Time': times})
test.to_excel('test.xlsx')