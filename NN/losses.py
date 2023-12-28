import numpy as np 

def mse(a,y):
    da = a - y
    loss = np.mean(np.sum(np.square(da), axis=0,keepdims=True))
    return loss, da

def softmax_loss(a,y):
    dz = a - y
    # For numerical stability of log calculation, near-zero values are brought up to a small value - epsilon
    epsilon = 1e-10
    log_a = np.log(np.maximum(a, epsilon))
    loss = -np.mean(np.sum(log_a *y, axis=0,keepdims=True))
    
    return loss, dz