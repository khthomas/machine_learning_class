import numpy as np
from scipy.io import loadmat
from sklearn import svm
import matplotlib.pyplot as plt
#from 
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

svdd = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)


svdd = svm.OneClassSVM(nu = 0.1, kernel="rbf", gamma=0.1)

#data = loadmat('/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/SVDD_Project/cardio.mat')
data = loadmat('/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/SVDD_Project/cardio.mat')

data2 = 

M = data['X']
L = data['y']

svdd.fit(M)

y_pred = svdd.fit(M).predict(M)

outliers = np.argwhere(y_pred == -1)
len(outliers)