from sklearn.datasets import fetch_openml
import pandas as pd


# get_mnist:  takes no arguments, connects over the internet to retrieve the
# MNIST digits data in the form of a matrix where rows are inputs, and a vector
# that contains the true label of the digit (0-9)
def get_mnist():
    examples, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    return examples, labels


X, y = get_mnist()

import numpy as np
import matplotlib.pyplot as plt

MNIST_WIDTH = 28


# show_mnist_index:  code to graphically display example i from matrix examples
# (which comes from get_mnist or similar)
def show_mnist_index(examples, i):
    # The colon operator selects "all" as the index, grabbing a whole row or column
    data_array = np.array(examples[i, :])
    # reshape takes the vector-like data and turns it into 2D, restoring the original shape
    image = np.reshape(data_array, (MNIST_WIDTH, MNIST_WIDTH))
    # Draw the image
    plt.imshow(image)
    # But matplotlib needs to be told to create the window or image
    plt.show()


from sklearn.neural_network import MLPClassifier


# trained_network:  returns a neural network object ("clf" for "classifier")
# that can learn via stochastic gradient descent using its .fit() method,
# then make predictions about new data with its .predict() method
def trained_network(examples, labels):
    clf = MLPClassifier(solver='sgd', learning_rate='constant', learning_rate_init=0.001, max_iter=400,
                        shuffle=True)
    clf.fit(examples, labels)
    return clf


# predict_mnist:  just a wrapper for a neural network's predict() function
# clf is the neural network returned earlier, examples is the example matrix from
# downloading the MNIST data, and i is an index to the ith example
def predict_mnist(clf, examples, i):
    return clf.predict([examples[i, :]])


print(predict_mnist(trained_network(X, y), X, 0))
print(predict_mnist(trained_network(X, y), X, 1))
print(predict_mnist(trained_network(X, y), X, 2))


# Append 1 for interaction with bias weight - this makes the calculation of the
# activations at the next layer a simple matrix multiplication
def append_1(x):
    return np.append(x, 1)


# Apply logistic (sigmoid) function to all elements of vector
def sigmoid(x):
    return [1 / (1 + np.exp(-i)) for i in x]


# Derivative of the sigmoid for all elements of the vector
def sigmoid_deriv(x):
    return [(1 / (1 + np.exp(-i)) * (1 - (1 / (1 + np.exp(-i))))) for i in x]


# See specifications on the inputs above
def predict(W_ij, W_jk, x):
    z1 = append_1(x)
    z2 = W_ij @ z1
    z3 = sigmoid(z2)
    z4 = append_1(z3)
    z5 = W_jk @ z4
    z6 = sigmoid([z5])
    return z6


# See return the desired weight matrices
def littlest():
    W_ij = np.array([[-1, -1, 1], [1, -1, 1], [0, 1, 1]])
    return W_ij, np.array([2, 2, 2, -5.4])


W_ij, W_jk = littlest()


# h is the index of the hidden element of interest
def plot_hidden(W_ij, h):
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-2, 2, 0.01)
    # Where x and y were 1D arrays of equally spaced values, meshgrid will
    # return a 2D grid of those values that try the various combinations of them;
    # this grid's x values are in the xx 2D array, and the y values are in the yy 2D array.
    # (For example, meshgrid on [0,1] and [0,1] returns [[0,1],[0,1]] and [[0,0],[1,1]],
    # corresponding to a grid containing (0,0), (1,0), (0,1), (1,1).)  This is handy for
    # working with 2D images that plot some function f(x,y) at each location.
    xx, yy = np.meshgrid(x, y)
    z = np.zeros((x.size, y.size))
    x_range, y_range = xx.shape
    for i in range(x_range):
        for j in range(y_range):
            z[i][j] = sigmoid([(W_ij[h])[0] * xx[i][j] + (W_ij[h])[1] * yy[i][j] + (W_ij[h])[2]])[0]
    h = plt.contourf(x, y, z)
    plt.show()


plot_hidden(W_ij, 0)
plot_hidden(W_ij, 1)
plot_hidden(W_ij, 2)


def plot_sol_output(W_ij, W_jk):
    x = np.arange(-2, 2, 0.01)
    y = np.arange(-2, 2, 0.01)
    # Where x and y were 1D arrays of equally spaced values, meshgrid will
    # return a 2D grid of those values that try the various combinations of them;
    # this grid's x values are in the xx 2D array, and the y values are in the yy 2D array.
    # (For example, meshgrid on [0,1] and [0,1] returns [[0,1],[0,1]] and [[0,0],[1,1]],
    # corresponding to a grid containing (0,0), (1,0), (0,1), (1,1).)  This is handy for
    # working with 2D images that plot some function f(x,y) at each location.
    xx, yy = np.meshgrid(x, y)
    z = np.zeros((x.size, y.size))
    x_range, y_range = xx.shape
    for i in range(x_range):
        for j in range(y_range):
            z[i][j] = predict(W_ij, W_jk, np.array([[xx[i][j]], [yy[i][j]]]))[0]
    h = plt.contourf(x, y, z)
    plt.show()


plot_sol_output(W_ij, W_jk)


# Get gradient of error as a prerequisite for backprop
# Returns gradient as two matrices, containing the derivatives with respect to
# each element of each weight matrix
# x is the input vector and y is the desired output
# Training set is 0 or 1, so we'll interpret the class as whichever is closer
def get_gradient(W_ij, W_jk, x, y):
    hidden_in = W_ij @ append_1(x)
    hidden_out = sigmoid(hidden_in)
    last_layer_in = W_jk @ append_1(hidden_out)
    output = sigmoid(last_layer_in)
    last_layer_derivs = sigmoid_deriv(last_layer_in)
    last_layer_delta = [-2 * (y[i] - output[i]) * last_layer_derivs[i] for i in range(len(output))]
    # outer product of delta and hidden transpose gives us what we want, a matrix of derivs
    # with rows for different output and cols for different hidden units
    dloss_dW2 = np.outer(last_layer_delta, append_1(hidden_out))
    hidden_layer_derivs = sigmoid_deriv(hidden_in)
    # A column of the second weight matrix corresponds to the outgoing signal
    # from a particular hidden unit - dot this with the final layer deltas
    # We can do this for each row with a transpose and matrix multiplication
    weight_dots = W_jk.T @ last_layer_delta
    hidden_layer_delta = [hidden_layer_derivs[i] * weight_dots[i] for i in range(len(hidden_layer_derivs))]
    # Again, taking the outer product of the vectors produces a matrix with entries in the right spots
    dloss_dW1 = np.outer(hidden_layer_delta, append_1(x))
    return dloss_dW1, dloss_dW2


# Print accuracy and loss for a given network described by matrices W_ij and W_jk, using
# the predict function to evaluate its results.  Rows of X are examples.
# y is a matrix with the desired output, nx1 in the case of a single classification output.
def print_accuracy_and_loss(W_ij, W_jk, X, y):
    total_right = 0
    loss = 0
    for i in range(len(y)):
        out = predict(W_ij, W_jk, X[i, :])
        if (out[0] < 0.5 and y[i][0] == 0):
            total_right += 1
        if (out[0] >= 0.5 and y[i][0] == 1):
            total_right += 1
        loss += (y[i][0] - out[0]) ** 2
    accuracy = total_right / len(y)
    print('Accuracy:')
    print(accuracy)
    print('Loss:')
    print(loss)
    return accuracy, loss


LEARN_RATE = 0.1


# Takes the two starting weight matrices, a matrix with each example a row,
# a vector of desired outputs, and a desired number of epochs to run.
# All vector or matrix args should be np.arrays.
# Prints accuracy and loss as we go, so we can check that we're learning something.
# Returns the learned weight matrices.
def backprop_learn(W_ij, W_jk, X, y, epochs):
    for i in range(epochs):
        m, n = np.shape(X)
        for j in range(m):
            grad1, grad2 = get_gradient(W_ij, W_jk, X[j, :], y[j])
            W_ij = W_ij - LEARN_RATE * grad1
            W_jk = W_jk - LEARN_RATE * grad2
        print('Epoch ' + str(i))
        print_accuracy_and_loss(W_ij, W_jk, X, y)
    return W_ij, W_jk


W_ij_rand = np.array([[-0.43740237, 0.27586301, 0.49627193],
                      [0.49250938, -0.02014223, 0.18785578],
                      [0.36062453, 0.28339736, 0.17899865],
                      [-0.30877367, 0.19374126, -0.36042027]])

W_jk_rand = np.array([[-0.37300732, 0.31031975, -0.24113355, 0.29412743, 0.44585899]])


def generate_data(n, max_abs):
    myvals = np.random.rand(n, 2)
    myvals *= (max_abs * 2)
    myvals -= max_abs
    # myvals are valid x values now; compute y
    y = np.zeros((n, 1))
    for i in range(n):
        if (abs(myvals[i][0]) + abs(myvals[i][1]) <= 1):
            y[i][0] = 1
    return myvals, y


Xabsdata, yabsdata = generate_data(100, 1)


# Given network matrices and matrix X of examples,
# plot how the network classifies them (red for yes, blue for no)
def plot_sol_results(W_ij, W_jk, X):
    X_height, _ = X.shape
    for i in range(X_height):
        if predict(W_ij, W_jk, X[i, :])[0] > 0.5:
            plt.scatter(X[i, 0], X[i, 1], c='#ff0000')
        else:
            plt.scatter(X[i, 0], X[i, 1], c='#0000ff')
    plt.show()


# This should look pretty bad at first, since the weights are random;
# final result should classify points near the center as red
plot_sol_results(W_ij_rand, W_jk_rand, Xabsdata)

# TODO:  use backprop to train W_ij_rand and W_jk_rand for 10000 epochs --+--
# and call plot_sol_results again
W_ij, W_jk = backprop_learn(W_ij_rand, W_jk_rand, Xabsdata, yabsdata, 10000)
plot_sol_results(W_ij, W_jk, Xabsdata)
