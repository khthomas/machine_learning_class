import numpy as np
from scipy.io import loadmat
from sklearn import svm

svdd = svm.OneClassSVM(nu = 0.1, kernel="rbf", gamma=0.1)

data = loadmat('/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/SVDD_Project/cardio.mat')

M = data['X']
L = data['y']

svdd.fit(M)

y_pred = svdd.fit(M).predict(M)

outliers = np.argwhere(y_pred == -1)
len(outliers)