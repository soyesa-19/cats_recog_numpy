import numpy as np

def sigmoid(z):

    s = 1./(1+np.exp(-z))

    return s

# print(sigmoid(np.array([0,2])))