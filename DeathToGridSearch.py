import numpy as np
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import seaborn
# adapt this code below to run your analysis
 
# Due before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each


# 
#  
#Due before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings #DONE
# 3. find some simple data #IRIS DATA WILL BE USED DONE
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings. THIS SHOULD BE A METHOD.
 
#Due before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function
 
# M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
# # L = np.ones(M.shape[0])
# L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])
# n_folds = 5
 
# data = (M, L, n_folds)

### Keep erroring out on data set, using IRIS data set for testing
iris = datasets.load_iris()
M = iris.data[:, :2] #taking the first two features, following example on sklearn website
L = iris.target
n_folds = 5
data = (M, L, n_folds)
 
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results
 
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
 
    clf.fit(M[train_index], L[train_index])
 
    pred = clf.predict(M[test_index])
 
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred),
               'predictions': pred}

  return ret
 
results = run(RandomForestClassifier, data, clf_hyper={})


testInput = [
{'clf': [LogisticRegression], 'C': [1.0], 'solver': ['lbfgs'], 'tol' : [1e-3]},
{'clf': [RandomForestClassifier]},
{'clf': [SVC], 'C': [1.1, 0.5], 'kernel': ['rbf'], 'tol': [1e-4], }
]


def unpackHypers(kwargs):
  #borrowed from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

# testInput =  [LogisticRegression, {'C': [0.5, 1.0, 1.5], 'solver': ['lbfgs', 'liblinear'], 'tol' : [1e-4, 1e-3, 1e-2]}]
def DeathToGridSearch(searchList):
# input is a list of lists with the format [model, hyperParamterdict], ideally the hyperparameters will be 
# expanded to accept a list of inputs right now it is just a single value to make sure variable 
# unpacking is passed to the classifer correctly

  for searchVar in searchList:
    allHypers = list(unpackHypers(searchVar))

    for search in allHypers:
          classifier = search.pop('clf', None) #if clf is a key pop it, return none otherwise
          hypers = search
          try:
          # some classifiers might fail, try to run it and if failure continue
            print('\n', classifier, hypers, '\n')
            res = run(classifier, data, hypers)
          except Exception as e:      
            res = f"Classifier {classifier} failed with the following error:\n{e}\nResuming Search."
        
          print(res)

DeathToGridSearch(testInput)




def plotResults(results_dict):
  """ Takes a results dictionary from run(). Generates plots to help evaluation. Plots have
  not yet been decided upon."""
  truth_data = M[results[0]['test_index']]
  truth_labels = L[results[0]['test_index']]
  pred_labels = results[0]['predictions']
  colors = ['b', 'g', 'r', 'c', 'm', 'y']


  # If there are 2 dimensions the we can do 2D Scatter Plots.
  if truth_data.ndim <= 2:
    x_vals = truth_data[:,0]
    y_vals = truth_data[:,1]
    titles = ["Truth", "Prediction"]import numpy as np
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import seaborn
# adapt this code below to run your analysis
 
# Due before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each


# 
#  
#Due before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings #DONE
# 3. find some simple data #IRIS DATA WILL BE USED DONE
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings. THIS SHOULD BE A METHOD.
 
#Due before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function
 
# M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
# # L = np.ones(M.shape[0])
# L = np.random.choice([0,1], size=(M.shape[0],), p=[1./3, 2./3])
# n_folds = 5
 
# data = (M, L, n_folds)

### Keep erroring out on data set, using IRIS data set for testing
iris = datasets.load_iris()
M = iris.data[:, :2] #taking the first two features, following example on sklearn website
L = iris.target
n_folds = 5
data = (M, L, n_folds)
 
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results
 
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist
 
    clf.fit(M[train_index], L[train_index])
 
    pred = clf.predict(M[test_index])
 
    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred),
               'predictions': pred}

  return ret
 
results = run(RandomForestClassifier, data, clf_hyper={})


testInput = [
{'clf': [LogisticRegression], 'C': [1.0], 'solver': ['lbfgs'], 'tol' : [1e-3]},
{'clf': [RandomForestClassifier]},
{'clf': [SVC], 'C': [1.1, 0.5], 'kernel': ['rbf'], 'tol': [1e-4], }
]


def unpackHypers(kwargs):
  #borrowed from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

# testInput =  [LogisticRegression, {'C': [0.5, 1.0, 1.5], 'solver': ['lbfgs', 'liblinear'], 'tol' : [1e-4, 1e-3, 1e-2]}]
def DeathToGridSearch(searchList):
# input is a list of lists with the format [model, hyperParamterdict], ideally the hyperparameters will be 
# expanded to accept a list of inputs right now it is just a single value to make sure variable 
# unpacking is passed to the classifer correctly

  for searchVar in searchList:
    allHypers = list(unpackHypers(searchVar))

    for search in allHypers:
          classifier = search.pop('clf', None) #if clf is a key pop it, return none otherwise
          hypers = search
          try:
          # some classifiers might fail, try to run it and if failure continue
            print('\n', classifier, hypers, '\n')
            res = run(classifier, data, hypers)
          except Exception as e:      
            res = f"Classifier {classifier} failed with the following error:\n{e}\nResuming Search."
        
          print(res)

DeathToGridSearch(testInput)




def plotResults(results_dict):
  """ Takes a results dictionary from run(). Generates plots to help evaluation. Plots have
  not yet been decided upon."""
  truth_data = M[results[0]['test_index']]
  truth_labels = L[results[0]['test_index']]
  pred_labels = results[0]['predictions']
  colors = ['b', 'g', 'r', 'c', 'm', 'y']


  # If there are 2 dimensions the we can do 2D Scatter Plots.
  if truth_data.ndim <= 2:
    x_vals = truth_data[:,0]
    y_vals = truth_data[:,1]
    titles = ["Truth", "Prediction"]
    labels= [truth_labels, pred_labels]

    for title, label in zip(titles, labels):
      seaborn.scatterplot(x=x_vals, y=y_vals, hue=label)

      # for lab in range(len(set(label))):
      #   this_color = colors[lab%len(colors)]
      #   plt.scatter(x_vals, y_vals, c=this_color, label=lab)
      #   plt.legend()
      #   plt.title(title)
      #   plt.show()

    
plotResults(results)



#testing area
clfTest = [
  [SVC, {'C': [1.0, 0.5], 'kernel': ['rbf', 'linear'], 'tol': [1e-4] }]
  ]


# # [LogisticRegression, {'C': 1.0, 'solver': 'lbfgs', 'tol' : 1e-3}]

# clf = clfTest[0][0]
# hyper = clfTest[0][1]
# run(clf, data, hyper)

# clfsList = [RandomForestClassifier, LogisticRegression] 

test = clfTest[0][1]

# def unpackHypers(kwargs):
#   #borrowed from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
#     keys = kwargs.keys()
#     vals = kwargs.values()
#     for instance in product(*vals):
#         yield dict(zip(keys, instance))


    labels= [truth_labels, pred_labels]

    for title, label in zip(titles, labels):
      seaborn.scatterplot(x=x_vals, y=y_vals, hue=label)

      # for lab in range(len(set(label))):
      #   this_color = colors[lab%len(colors)]
      #   plt.scatter(x_vals, y_vals, c=this_color, label=lab)
      #   plt.legend()
      #   plt.title(title)
      #   plt.show()

    
plotResults(results)



#testing area
clfTest = [
  [SVC, {'C': [1.0, 0.5], 'kernel': ['rbf', 'linear'], 'tol': [1e-4] }]
  ]


# # [LogisticRegression, {'C': 1.0, 'solver': 'lbfgs', 'tol' : 1e-3}]

# clf = clfTest[0][0]
# hyper = clfTest[0][1]
# run(clf, data, hyper)

# clfsList = [RandomForestClassifier, LogisticRegression] 

test = clfTest[0][1]

# def unpackHypers(kwargs):
#   #borrowed from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
#     keys = kwargs.keys()
#     vals = kwargs.values()
#     for instance in product(*vals):
#         yield dict(zip(keys, instance))

