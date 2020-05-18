import numpy as np
import h5py
import matplotlib.pyplot as plt
import my_package


# Loading the dataset
# x=images, y=label
train_dataset = h5py.File("dataset/train_catvnoncat.h5",'r')   #training set
train_x = np.array(train_dataset["train_set_x"][:])  #(209,64,64,3)
train_y = np.array(train_dataset["train_set_y"][:])  #(209,)


test_dataset = h5py.File("dataset/test_catvnoncat.h5",'r')   #testing set
test_x = np.array(test_dataset["test_set_x"][:])   #(50,64,64,3)
test_y = np.array(test_dataset["test_set_y"][:])   #(50,)

classes = np.array(test_dataset["list_classes"][:])

train_y = train_y.reshape((1, train_y.shape[0]))  #converting (209,) to (1,209)
test_y = test_y.reshape((1, test_y.shape[0]))     #converting (50,) to (1,50)


no_of_train_img = train_x.shape[0]
no_of_test_img = test_x.shape[0]
size_of_img = train_x.shape[1]

# reshape train and test images
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

# standardizing our dataset
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.



# training the model
d= my_package.models.model(train_x, train_y, test_x, test_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

