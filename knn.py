import numpy as np 


#fonction generale pour knn  ---by dia --
def K_nn(X,Y,attribute,k) :
    distances = np.zeros((X.shape[0],),dtype=float32)
    i=0
    for x in X :
        distances[i] = np.square(np.linalg.norm(x - attribute))
        i+=1
    min_val_indexes = []
    for i in range(k):
        minimum = np.min(ditances)
        index = distances.index(minium)
        min_val_indexes.append(index)
        distances[index] = float('Inf')
    final_classes = {}
    for idx in min_val_indexes :
        if(Y[idx] not in final_classes.keys()):
            final_classes[Y[idx]] = 1
        else:
            final_classes[Y[idx]] += 1
    occ = [val for val in final_classes.values()]
    maximum = np.max(occ)
    for key , val in final_classes.items():
        if(val == maximum):
            yhat = key
            break
    
    return yhat