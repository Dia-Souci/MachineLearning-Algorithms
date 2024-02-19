import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

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

def executeDBSCAN(X,eps,min_samples):
    eps = float(eps)
    min_samples = int(min_samples)
    top = tk.Toplevel()
    top.geometry("600x400") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.update()
    top.title('Details')
    top['bg']='#EFF5F5'
    insideFrame1 = tk.LabelFrame(top,text='Display')
    insideFrame1.place(height=400, width=600 , rely=0.05 ,relx=0)
    insideFrame1['bg']='#D6E4E5'
    text = tk.Text(insideFrame1, height=400, width=600)
    scroll = tk.Scrollbar(insideFrame1)
    text.configure(yscrollcommand=scroll.set)
    text.place( rely=0.05 ,relx=0)
    insert_text = ""
    insert_text += "For dataset1 : \n"
    insert_text += "eps = %f , min_samples = %d \n" % (eps,min_samples)

    start = time.time()
    labels , n_clusters_ , n_noise_ = DBSCAN_scratch(X,eps,min_samples)
    end = time.time()
    insert_text += "Execution time: %f \n" % (end-start)
    insert_text += 'Estimated number of clusters: %d \n' % (n_clusters_) 
    print("Execution time: %f" % (end-start))
    print('Estimated number of clusters: %d' % n_clusters_)
    score = silhouette_score(X, labels,metric='euclidean')
    insert_text += "Silhouette Coefficient: %0.3f \n" % score
    print("Silhouette Coefficient: %0.3f" % score)
    insert_text += 'Estimated number of noise points: %d \n' % (n_noise_)
    print('Estimated number of noise points: %d' % n_noise_)
    text.insert(tk.END, insert_text)

    return

def executeAgglomerative(n_clusters):
    n_clusters = int(n_clusters)
    top = tk.Toplevel()
    top.geometry("600x400") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.update()
    top.title('Details')
    top['bg']='#EFF5F5'
    insideFrame1 = tk.LabelFrame(top,text='Display')
    insideFrame1.place(height=400, width=600 , rely=0.05 ,relx=0)
    insideFrame1['bg']='#D6E4E5'
    text = tk.Text(insideFrame1, height=400, width=600)
    scroll = tk.Scrollbar(insideFrame1)
    text.configure(yscrollcommand=scroll.set)
    text.place( rely=0.05 ,relx=0)
    insert_text = ""
    insert_text += "For dataset1 : \n"

    start = time.time()
    clusters = agglomerative_clustering(X, n_clusters)
    end = time.time()
    insert_text += "Execution time: %f \n" % (end-start)
    insert_text += "number of clusters: %d \n" % len(clusters)
    labels = np.zeros(len(X))
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            labels[clusters[i][j]] = i
    if(len(clusters)> 1):
        score = silhouette_score(X, labels,metric='euclidean')
        insert_text += "Silhouette Coefficient: %0.3f \n" % score
        print("Silhouette Coefficient: %0.3f" % score)
    text.insert(tk.END, insert_text)
    return


root = tk.Tk()

root.geometry("800x650") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.update()

root.title("Clustering Methods")
root['bg'] = '#EFF5F5'

frame1 = tk.LabelFrame(root,text='DBSCAN')
frame1.place(height=300, width=700, rely=0.05 ,relx=0.03)
frame1['bg'] ='#D6E4E5'

label1 = tk.Label(frame1,text="Epsilon")
label1.place(rely=0.1 , relx = 0.01)

label2 = tk.Label(frame1,text="Min Samples")
label2.place(rely=0.3 , relx = 0.01)

Entry1 = tk.Entry(frame1)
Entry1.place(rely=0.3 , relx = 0.6,width=200)

Entry = tk.Entry(frame1)
Entry.place(rely=0.1 , relx = 0.6,width=200)

dbscan_btn = tk.Button(frame1,text="Apply DBSCAN", command = lambda : executeDBSCAN(X,Entry.get(),Entry1.get()))
dbscan_btn.place(rely=0.6 , relx = 0.3,width=200)
dbscan_btn['bg']='#EFF5F5'



frame2 = tk.LabelFrame(root,text='Agglomerative')
frame2.place(height=250, width=700, rely=0.55 ,relx=0.03)
frame2['bg'] ='#D6E4E5'

label3 = tk.Label(frame2,text="Number of Clusters")
label3.place(rely=0.1 , relx = 0.01)

Entry2 = tk.Entry(frame2)
Entry2.place(rely=0.1 , relx = 0.6,width=200)

agnes_btn = tk.Button(frame2,text="Apply AGNES", command = lambda : executeAgglomerative(Entry2.get()))
agnes_btn.place(rely=0.6 , relx = 0.3,width=200)
agnes_btn['bg']='#EFF5F5'


root.mainloop()