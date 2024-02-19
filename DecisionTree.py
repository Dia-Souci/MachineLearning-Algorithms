import numpy as np
import matplotlib.pyplot as plt


def compute_entropy(y):
    
    # You need to return the following variables correctly
    entropy = 0.
    
    ### START CODE HERE ###
    m = len(y)
    if (m !=0):
        p1= np.sum(y) / m
        if(p1 == 1 or p1 == 0 ):
            entropy = 0.
        else:
            entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
    
    ### END CODE HERE ###        
    
    return entropy

def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    
    ### START CODE HERE ###
    for index in node_indices :
        if(X[index][feature] == 1):
            left_indices.append(index)
        else : 
            right_indices.append(index)
     
    ### END CODE HERE ###
        
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0
    
    
    
    ### START CODE HERE ###
    
    H_node = compute_entropy(y_node)
    H_left = compute_entropy(y_left)
    H_right = compute_entropy(y_right)
    w_left = len(X_left)/len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_H = w_left * H_left + w_right * H_right
    
    information_gain = H_node - weighted_H
    ### END CODE HERE ###  
    
    return information_gain

def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1
    ind = -1
    
    ### START CODE HERE ###
    feat= []
    for index in range(num_features) :
        value = compute_information_gain(X, y, node_indices,index)
        feat.append(value)
    maxV = np.max(feat)
    if(maxV > 0 ) :
        best_feature = feat.index(maxV)
    ### END CODE HERE ##    
   
    return best_feature

tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)