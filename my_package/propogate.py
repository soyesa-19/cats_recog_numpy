import numpy as np
import my_package.sigmoid

def propagate(w,b,X,Y):

    m= X.shape[1]

    # forward propogation
    A = my_package.sigmoid.sigmoid(np.dot(w.T,X) + b)
    cost = (-1. / m) * np.sum((Y*np.log(A) + (1 - Y)*np.log(1-A)), axis=1)

    # backward propogation
    dw = (1./m)*np.dot(X,((A-Y).T))
    db = (1./m)*np.sum(A-Y, axis=1)


    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

# w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
# grads, cost = propagate(w, b, X, Y)
# print(grads, cost)