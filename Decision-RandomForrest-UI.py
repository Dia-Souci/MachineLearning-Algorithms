import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
import time


DATA_PATH = os.path.join('.','data')
ORIGINAL_PATH = os.path.join(DATA_PATH, 'Data1Reduced.xlsx')

global df

df= pd.read_excel(ORIGINAL_PATH)
Y = df['Attrition'].to_numpy()
X = df.loc[:,df.columns != 'Attrition'].to_numpy()

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

class Node():
    def __init__(self, feature_index=None,left=None, right=None, value=None):
        self.feature_index = feature_index
        self.left = left
        self.right = right
        self.value = value

def drawConfusionMatrix(Y_pred,Y_true):
    matrix = np.zeros((2,2))
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(len(Y_pred)):
        if(Y_true[i]==1):
            if(Y_pred[i]==1):
                tp += 1
            else:
                fn +=1
        else:
            if(Y_pred[i]==1):
                fp += 1
            else:
                tn +=1
    matrix[0][0]=tp
    matrix[0][1]=fp
    matrix[1][0]=fn
    matrix[1][1]=tn
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    sensitivity =precision
    specificity = tn / (tn+fp)
    f1_score = 2*(precision*recall)/(precision+recall)
    return matrix,accuracy,precision,recall,sensitivity,specificity,f1_score


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

def applyThresh(Y,indices):
    tot=0
    # sprint(max(Y[indices]))
    for x in indices :
        tot += Y[x]
    if (tot/len(indices)<=0.15):
        return 0
    else :
        return 1 

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    best_feature = get_best_split(X, y, node_indices)

    if (current_depth == max_depth)or(best_feature == -1 ):
        formatting = " "*current_depth + "-"*current_depth
        # print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        Value = applyThresh(y,node_indices)
        return Node(value=Value)
   
     
    
    formatting = "-"*current_depth
    # print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature

    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    # continue splitting the left and the right child. Increment current depth
    left_sub = build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    right_sub = build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
    return Node(feature_index= best_feature,left=left_sub,right=right_sub)

def BuildDecisionTree(X, y, node_indices, branch_name, max_depth, current_depth):
    start = time.time()
    tree =build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth)
    end = time.time()
    execution = end-start
    return tree , execution

def predict(X,Node):
    while(Node.value == None):
        if(X[Node.feature_index]==1):
            Node=Node.left
        else :
            Node=Node.right
    return Node.value

def generatePrediction(X,Node):
    Y_pred =np.array([predict(element,Node) for element in X])
    return Y_pred
        
def GenerateData(numberOfsamples,portion):
    data = []
    targets =[]
    for i in range(numberOfsamples):
        newdf = df.sample(n=portion,replace=True)
        Y = newdf['Attrition'].to_numpy()
        X = newdf.loc[:,newdf.columns != 'Attrition'].to_numpy()
        data.append(X)
        targets.append(Y)
    return data,targets

def buildForest (NumberOfTrees):
    data,targets = GenerateData(NumberOfTrees,750)
    root_indices = [i for i in range(data[0].shape[0])]
    start = time.time()
    forest = [build_tree_recursive(data[e], targets[e], root_indices, "Root", max_depth=10, current_depth=0) for e in range(NumberOfTrees) ]
    end = time.time()
    execution = end-start

    return forest , execution

def predictForest(X,forest):
    somme =0
    for Node in forest :
        while(Node.value == None):
            if(X[Node.feature_index]==1):
                Node=Node.left
            else :
                Node=Node.right
        somme += Node.value
    if (somme / len(forest)) >= 0.5 :
        return 1
    else :
        return 0

def generatePredectionForest(X,forest):
    Y_pred =np.array([predictForest(element,forest) for element in X])
    return Y_pred
        

def ApplyTree():
    root_indices = [i for i in range(X_train.shape[0])]
    global tree
    tree , timeTree = BuildDecisionTree(X_train, Y_train, root_indices, "Root", max_depth=10, current_depth=0)
    Y_pred_tree =generatePrediction(X_test,tree)
    matrix , accuracy,precision , recall,sens,spec,f1_score = drawConfusionMatrix(Y_pred_tree,Y_test)
    top = tk.Toplevel()
    top.geometry("600x400") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.resizable(0, 0) # makes the root window fixed in size.
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
    insert_text += "Results For 1 Decision tree : \n"
    insert_text += "Execution Time :  " + str(timeTree) + "\n"
    insert_text += "Accuracy :  " + str(accuracy) + "\n"
    insert_text += "F1-Score :  " + str(f1_score) + "\n"
    insert_text += "recall :  " + str(recall) + "\n"
    insert_text += "Specificity :  " + str(spec) + "\n"
    insert_text += "Confusion Matrix :  \n"
    insert_text+=  str(matrix) + "\n"
    text.insert(tk.END, insert_text)
    return

#1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0

#1,0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0

ref =[]

def ApplyForest():
    global forest
    forest , timeForest = buildForest(7)
    Y_pred = generatePredectionForest(X_test,forest)

    matrix , accuracy,precision , recall,sens,spec,f1_score = drawConfusionMatrix(Y_pred,Y_test)
    top = tk.Toplevel()
    top.geometry("600x400") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.resizable(0, 0) # makes the root window fixed in size.
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

    insert_text += "Results For Random Forest with 7 trees : \n"
    insert_text += "Execution Time :  " + str(timeForest) + "\n"
    insert_text += "Accuracy :  " + str(accuracy) + "\n"
    insert_text += "F1-Score :  " + str(f1_score) + "\n"
    insert_text += "recall :  " + str(recall) + "\n"
    insert_text += "Specificity :  " + str(spec) + "\n"
    insert_text += "Confusion Matrix :  \n" 
    insert_text+= str(matrix) + "\n"
    text.insert(tk.END, insert_text)
    return


def PredictForestInterface(data):
    dataPrint = [e.get() for e in ref]
    print(forest[0].feature_index)
    y_hat = predictForest(data,forest)
    y_name = ["Attrition : No","Attrition : Yes"]
    top = tk.Toplevel()
    top.geometry("600x400") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.resizable(0, 0) # makes the root window fixed in size.
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
    insert_text += "For data : \n"
    insert_text += str(dataPrint) + '\n'
    insert_text +="Prediction : "+str(y_name[y_hat])
    text.insert(tk.END, insert_text)
    return

def PredictTreeInterface(data):
    dataPrint = [e.get() for e in ref]
    print(tree.feature_index)
    y_hat = predict(data,tree)
    y_name = ["Attrition : No","Attrition : Yes"]
    top = tk.Toplevel()
    top.geometry("600x400") # set the root dimensions
    top.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
    top.resizable(0, 0) # makes the root window fixed in size.
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
    insert_text += "For data : \n"
    insert_text += str(dataPrint) + '\n'
    insert_text +="Prediction : "+str(y_name[y_hat])
    text.insert(tk.END, insert_text)
    return

def GenData(ref,textLabels):
    data = [e.get() for e in ref]
    global form
    form=np.zeros(X_train.shape[0])
    for i in range(len(textLabels)):
        if (i==0):
            if(int(data[0])<30):
                form[3]=1
                form[4]=0
                form[5]=0
                form[6]=0
            elif(int(data[0])<36):
                form[3]=0
                form[4]=1
                form[5]=0
                form[6]=0
            elif(int(data[0])<43):
                form[3]=0
                form[4]=0
                form[5]=1
                form[6]=0
            else:
                form[3]=0
                form[4]=0
                form[5]=0
                form[6]=1
        elif(i==1):
            if(data[1]=="Non-Travel"):
                form[7]=1
                form[8]=0
                form[9]=0
            elif(data[1]=="Travel_Frequently"):
                form[7]=0
                form[8]=1
                form[9]=0
            else:
                form[7]=0
                form[8]=0
                form[9]=1
        elif(i==2):
            if(int(data[2])<466):
                form[10]=1
                form[11]=0
                form[12]=0
                form[13]=0
            elif(int(data[2])<805):
                form[10]=0
                form[11]=1
                form[12]=0
                form[13]=0
            elif(int(data[2])<1158):
                form[10]=0
                form[11]=0
                form[12]=1
                form[13]=0
            else:
                form[10]=0
                form[11]=0
                form[12]=0
                form[13]=1
        elif(i==3):
            if(data[i]=="Human Resources"):
                form[14]=1
                form[15]=0
                form[16]=0
            elif(data[i]=="Sales"):
                form[14]=0
                form[15]=0
                form[16]=1
            else:
                form[14]=0
                form[15]=1
                form[16]=0
        elif(i==4):
            if(int(data[i])<2):
                form[17]=1
                form[18]=0
                form[19]=0
                form[20]=0
            elif(int(data[i])<7):
                form[17]=0
                form[18]=1
                form[19]=0
                form[20]=0
            elif(int(data[i])<14):
                form[17]=0
                form[18]=0
                form[19]=1
                form[20]=0
            else:
                form[17]=0
                form[18]=0
                form[19]=0
                form[20]=1
        elif(i==5):
            if(int(data[i])<2):
                form[21]=1
                form[22]=0
                form[23]=0
                form[24]=0
            elif(int(data[i])<3):
                form[21]=0
                form[22]=1
                form[23]=0
                form[24]=0
            elif(int(data[i])<4):
                form[21]=0
                form[22]=0
                form[23]=1
                form[24]=0
            else:
                form[21]=0
                form[22]=0
                form[23]=0
                form[24]=1
        elif(i==6):
            if(int(data[i])==1):
                form[25]=1
                form[26]=0
                form[27]=0
                form[28]=0
            elif(int(data[i])==2):
                form[25]=0
                form[26]=1
                form[27]=0
                form[28]=0
            elif(int(data[i])==3):
                form[25]=0
                form[26]=0
                form[27]=1
                form[28]=0
            else:
                form[25]=0
                form[26]=0
                form[27]=0
                form[28]=1
        elif(i==7):
            if(data[i]=="Female"):
                form[0]=0
            else:
                form[0]=1    
        elif(i==8):
            init = 29
            if(int(data[i])<48):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<66):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<84):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==9):
            init = 33
            if(int(data[i])==1):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==2):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==3):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==10):
            init = 37
            if(int(data[i])<2):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
            elif(int(data[i])<3):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=1
        elif(i==11):
            init = 40
            if(int(data[i])==1):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==2):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==3):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==12):
            init =44
            if(data[i]=="Divorced"):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
            elif(data[i]=="Married"):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=1
        elif(i==13):
            init = 47
            if(int(data[i])<2929):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<4941):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<8392):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==14):
            init = 51
            if(int(data[i])<8045):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<14242):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<20445):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==15):
            init = 55
            if(int(data[i])<1):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<2):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<4):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==16):
            if(data[i]=="No"):
                form[1]=0
            else:
                form[1]=1  
        elif(i==17):
            init = 59
            if(int(data[i])<12):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<14):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<18):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==18):
            if(data[i]==3):
                form[2]=0
            else:
                form[2]=1  
        elif(i==19):
            init = 63
            if(int(data[i])==1):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==2):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==3):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==20):
            init = 67
            if(int(data[i])==0):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==1):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==2):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==21):
            init = 71
            if(int(data[i])<6):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<10):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<15):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==22):
            init = 75
            if(int(data[i])<2):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
            elif(int(data[i])<3):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
            elif(int(data[i])<6):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
        elif(i==23):
            init = 78
            if(int(data[i])==1):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==2):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])==3):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==24):
            init = 82
            if(int(data[i])<3):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<5):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<9):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==25):
            init = 86
            if(int(data[i])<2):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<3):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<7):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        elif(i==26):
            init = 90
            if(int(data[i])<1):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
            elif(int(data[i])<3):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=1
        elif(i==27):
            init = 93
            if(int(data[i])<2):
                form[init]=1
                form[init+1]=0
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<3):
                form[init]=0
                form[init+1]=1
                form[init+2]=0
                form[init+3]=0
            elif(int(data[i])<7):
                form[init]=0
                form[init+1]=0
                form[init+2]=1
                form[init+3]=0
            else:
                form[init]=0
                form[init+1]=0
                form[init+2]=0
                form[init+3]=1
        print("Done")
        print(form)

    return form
#------------------------------------------------IHM----------------------------------------------------------------

root = tk.Tk()

root.geometry("800x650") # set the root dimensions
root.pack_propagate(False) # tells the root to not let the widgets inside it determine its size.
root.resizable(0, 0) # makes the root window fixed in size.
root.update()

root.title("Random Forest & Decision Trees")
root['bg'] = '#EFF5F5'

frame1 = tk.LabelFrame(root,text='Choosing Method')
frame1.place(height=200, width=250, rely=0.05 ,relx=0.03)
frame1['bg'] ='#D6E4E5'

Tree_btn = tk.Button(frame1,text="Apply Decision Tree", command = lambda : ApplyTree())
Tree_btn.place(rely=0.2 , relx = 0.1,width=200)
Tree_btn['bg']='#EFF5F5'

Forest_btn = tk.Button(frame1,text="Apply Random Forest", command = lambda : ApplyForest())
Forest_btn.place(rely=0.6 , relx = 0.1,width=200)
Forest_btn['bg']='#EFF5F5'

frame2 = tk.LabelFrame(root,text='Classify')
frame2.place(height=250, width=250, rely=0.55 ,relx=0.03)
frame2['bg'] ='#D6E4E5'



Tree_btn2 = tk.Button(frame2,text="Predict with Decision Tree", command = lambda : PredictTreeInterface(form))
Tree_btn2.place(rely=0.2 , relx = 0.1,width=200)
Tree_btn2['bg']='#EFF5F5'

Forest_btn2 = tk.Button(frame2,text="Predict with Random Forest", command = lambda : PredictForestInterface(form))
Forest_btn2.place(rely=0.6 , relx = 0.1,width=200)
Forest_btn2['bg']='#EFF5F5'

frame_data_entries = tk.LabelFrame(root,text='Data Insertion')
frame_data_entries.place(height=600, width=450, rely=0.05 ,relx=0.4)
frame_data_entries['bg'] ='#D6E4E5'

textLabels = ["Age","Travel","DailyRate","Department","Distance","Education","Envir.Satis","Gender","HourlyRate","JobInv","JobLevel","JobSatis","MaritalStatus","MonthInc","MonthRate","NumCompanies","OverTime","SalaryHike","PerfeRate","Rel.Satis","StockLevel","TotYears","Train-LastYear","Work-Life","Years-Comp","Years-currRole","Y-LastPromotion","Y-CurrManager"
]
print(len(textLabels))
x=0.01
x2=0.015
y=0.05
y2=0.3
for e in textLabels :
    if(x>0.85):
        x=0.01
        x2=0.015
        y=0.55
        y2=0.75
    Label1_= tk.Label(frame_data_entries,text=e)
    Label1_.place(rely=x , relx = y) 
    Label1_['bg']='#D6E4E5'

    Entry1__ = tk.Entry(frame_data_entries,width=15 ,font=('Arial 8'))
    Entry1__.place(rely=x2,relx=y2)
    Entry1__.insert(0,"Input")
    ref.append(Entry1__)
    x+=0.06
    x2+=0.06
    
genData_btn =tk.Button(frame_data_entries,text="Generate Data", command = lambda : GenData(ref,textLabels))
genData_btn.place(rely=0.9 , relx = 0.3 ,width=200)
genData_btn['bg']='#EFF5F5'

root.mainloop()



#

#-----------------------------------------------------IHM----------------------------------------------------------