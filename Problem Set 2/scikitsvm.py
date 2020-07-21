import numpy as np
import csv
from sklearn import svm


def load_csv(csv_path, label_col ='y'):
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    y_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter = ',', skiprows = 1, usecols = x_cols)
    labels = np.loadtxt(csv_path, delimiter = ',', skiprows = 1, usecols = y_cols)

    return inputs,labels
        
train_x, train_y = load_csv('../data/ds5_train.csv')
test_x, test_y = load_csv('../data/ds5_test.csv')

rbf = svm.SVC(kernel='rbf',gamma = 1)

rbf.fit(train_x,train_y)

y_predict = rbf.predict(test_x)

np.savetxt('./output/p05_rbfscikit_predictions',y_predict)
print(rbf.score(test_x, test_y))


