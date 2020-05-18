import numpy as np

def initialize_with_zeros(dim):

    w=np.zeros([dim,1], dtype=np.float32)
    b = 0 

    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w,b


# w,b = initialize_with_zeros(2)
# print(w)
# print(b)