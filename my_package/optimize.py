import numpy as np
import my_package.propogate

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs=[]

    for i in range(num_iterations):

        grads,cost = my_package.propogate.propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,"b": b}
    
    grads = {"dw": dw,"db": db}

    return params,grads,costs

