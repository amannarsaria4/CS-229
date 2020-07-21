# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        #if i % 10000 == 0:
            #print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return

def plot(x,y,save_path):
    
    #plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'ro', linewidth=2)
    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)

    

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    logistic_regression(Xa, Ya)
    plot(Xa,Ya,'output/data_a.png')
    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    #logistic_regression(Xb, Yb)
    plot(Xb,Yb,'output/data_b.png')

if __name__ == '__main__':
    main()
