import numpy as np
import pandas as pd
import os
import pprint
#from Part1_2 import simplify_
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
#---------------------------------------------------Data Preprocessing (in another file) -----------------------------------------

DATA_PATH = os.path.join('.','data')
ORIGINAL_PATH = os.path.join(DATA_PATH, 'Data1Reduced.xlsx')

global df
Original_df= pd.read_excel(ORIGINAL_PATH)
df = Original_df.copy()
Ydf = df['Attrition']
df.drop('Attrition',inplace=True,axis=1)
cols = df.columns
X = df.to_numpy()
Y = Ydf.to_numpy()



X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.33,random_state=42)
# print(len(X_train),len(X_test),len(Y_train),len(Y_test))


#---------------------------------------------------------Decision Tree --------------------------------------


def compute_entropy(y):
    entropy = 0.
    m = len(y)
    if (m !=0):
        p1= np.sum(y) / m
        if(p1 == 1 or p1 == 0 ):
            entropy = 0.
        else:
            entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
    
    return entropy

def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []

    for index in node_indices :
        if(X[index][feature] == 1):
            left_indices.append(index)
        else : 
            right_indices.append(index)
     
        
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    
    H_node = compute_entropy(y_node)
    H_left = compute_entropy(y_left)
    H_right = compute_entropy(y_right)
    w_left = len(X_left)/len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_H = w_left * H_left + w_right * H_right
    
    information_gain = H_node - weighted_H
    
    return information_gain

def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
   
    best_feature = -1

    feat= []
    for index in range(num_features) :
        value = compute_information_gain(X, y, node_indices,index)
        feat.append(value)
    maxV = np.max(feat)
    if(maxV > 0 ) :
        best_feature = feat.index(maxV)
  
    return best_feature


i=0
def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth,father):
    best_feature = get_best_split(X, y, node_indices)

    if (current_depth == max_depth)or(best_feature == -1 ):
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
     
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature

    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature,current_depth,father))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1,father+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1,father+1)

tree = []
root_indices =[]
for i in range (X_train.shape[0]):
    root_indices.append(i)

build_tree_recursive(X_train, Y_train, root_indices, "Root", max_depth=X_train.shape[1], current_depth=0,father=0)
inst = X_test[27]
 
print(tree[0][3])
idxs=[]
idxs.append(0)

def lookUpLastChild(X, tree , father) :
    idx = -10
    if (X[tree[father][2]]==1):
        left =True
        newFather=tree[father][3]
    else :
        left = False
        newFather=tree[father][3]

    found = False
    for x in tree :
        if(tree.index(x)>father):
            if(x[3]==newFather+1):
                if(left == True):
                    idx= tree.index(x)
                    # print(idx)
                    found =True
                else :
                    left = True
            if(found == True ):
                break
    if(idx > 0 ):
        idxs.append(idx)
        lookUpLastChild(X,tree,idx)

# lookUpLastChild(inst,tree,0)
# print(idxs)
def thresHoldVote(K,Y,indices):
    array =np.zeros(K)
    for e in range(K) :
        value = applyThresh(Y,indices)
        array[e]=value
    mean = np.mean(array)
    if(mean>0.5):
        return 1
    else :
        return 0

def GenerateForest (K):
    forest=[]
    for i in range(K) :
        newdf = Original_df.sample(n=900,replace=True)
        newYdf = newdf['Attrition']
        newdf.drop('Attrition',inplace=True,axis=1)
        X_ = newdf.to_numpy()
        Y_ = newYdf.to_numpy()
        tree = []
        root_indices = []
        for index in range(X_.shpae[0]):
            root_indices.append(index)
        build_tree_recursive(X_,Y_,root_indices,"root",X_.shape[0],0,0)
        forest.append(tree)
    return forest


def applyThresh(Y,indices):
    tot=0
    # sprint(max(Y[indices]))
    for x in indices :
        tot += Y[x]
    if (tot/len(indices)<=0.15):
        return 0
    else :
        return 1 
def fit (X , Y,tree ):

    y_name = ["Attrition : No","Attrition : Yes"]
    idxs = []
    idxs.append(0)
    lookUpLastChild(X,tree,0)
    if(X[tree[idxs[-1]][2]]==1):
        indices = tree[idxs[-1]][0]
    else:
        indices = tree[idxs[-1]][1]
    vaL =thresHoldVote(1,Y,indices)
    # print(" for this attribute the class is %s" % y_name[vaL])
    return vaL

    # break

def fitSet(X):
    Y_hat=[]
    for x in X :
        Y_hat.append(fit(x,Y_train,tree))
    return Y_hat

y_test_pred =fitSet(X_test)
y_train_pred = fitSet(X_train)

def calculateAccuracy(Y,Y_true):
    total =0
    for i in range(len(Y)):
        if (Y[i]==Y_true[i]):
            total += 1
    if(len(Y) != 0):
        return total/len(Y)
    else:
        return 0

arr = [22, 47, 52, 63, 72, 80, 89, 102, 111, 151, 235, 256, 272, 275, 279, 311, 325, 349, 359, 414, 421, 
430, 451, 470, 477, 488, 524, 545, 561, 574, 598, 611, 616, 639, 651, 663, 668, 670, 687, 691, 693, 701, 727, 759, 818, 841, 844, 858, 860, 862, 872, 884, 897, 942, 960]

arr2=[34, 239, 487, 784]

for x in arr :
    print(Y_train[x])

print("Accuracy in train :")
print(calculateAccuracy(y_train_pred,Y_train))
print(precision_recall_fscore_support(Y_train,y_train_pred,average='micro'))
print("Accuracy in test :")
print(calculateAccuracy(y_test_pred,Y_test))
print(precision_recall_fscore_support(Y_test,y_test_pred,average='micro'))

#----------------------------------- random forest---------------------------------------------------

