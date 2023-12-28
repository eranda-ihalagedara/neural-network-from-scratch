import numpy as np

def relu(x):
    return np.maximum(0,x)


def relu_prime(x):
    return (x>=0).astype(int)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    sig = sigmoid(x)
    return sig*(1-sig)


def linear(x):
    return x


def linear_prime(x):
    return np.ones(x.shape)


def softmax(x):

    # For numerical stability the max value of each vector is subtracted from all the elements
    z = x - np.max(x,axis=0,keepdims=True)
    ex = np.exp(z)
    sm = np.sum(ex, axis=0, keepdims=True)
    
    return ex/sm