import numpy as np

def intilization(X, Y):
    """_summary_

    Args:
        X (Data set): the features (m, n)
        Y (Data set): the target (m, 1)
    """
    
    n = X.shape[0]
    
    W = np.random.rand(1,n)
    b = 0
    return W,b

def fit(X, Y, alpha, n_iteration, lambd):
    W, b = intilization(X,Y)
    m, n = X.shape
    costs = []
    costr = 0
    c = np.sum(Y)
    for iter in range(n_iteration):
        
        costr = cost(W, b, X, Y, alpha, lambd)
        W -= alpha * (2 * lambd * W - np.dot(X, Y)) # w = w - α* (2λw - yixi)
        b -= alpha * Y # b = b - α* (yi)
        if(iter % 1000 == 0):
            print("cost at ", iter, " = ", costr)
            costs.append(costr)
    return W, b, costs

def predict(W,b, X):
    a3 = (np.dot(W,X) - b)
    p = np.random.rand(a3.shape[0])*0
    
    for i in range(0, a3.shape[1]):
            if a3[0,i] > 0:
                p[0,i] = 1
            else:
                p[0,i] = -1
    
    return p

def cost(W, b, X, Y, alpha, lambd):
    cost =  alpha * lambd * np.sqrt(np.sum(np.square(W))) + np.maximum(0, np.sum(1 - Y * ( np.dot(W, X ) - b ) ))
    return cost


