import numpy as np
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# adapt this code below to run your analysis
 
# Due before live class 2
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression, each with 3 different sets of hyper parrameters for each

# input is a list with format [model, hyperParamterdict], ideally the hyperparameters will be expanded to accept a list of inputs
# right now it is just a single value to make sure variable unpacking is passed to the classifer correctly
testInput = [
[LogisticRegression, {'C': 1.0, 'solver': 'lbfgs', 'tol' : 1e-3}]
]
# testInput =  [LogisticRegression, {'C': [0.5, 1.0, 1.5], 'solver': ['lbfgs', 'liblinear'], 'tol' : [1e-4, 1e-3, 1e-2]}]
def DeathToGridSearch(data, searchList):
  M, L, n_folds = data

  for searchVar in searchList:
    clf = searchVar[0]
    clf = clf(**searchVar[1])
    clf.fit(M,L)
    





 
#Due before live class 3
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
 
#Due before live class 4
# 5. Please set up your code to be run and save the results to the directory that its executed from
# 6. Investigate grid search function
 
M = np.array([[1,2],[3,4],[4,5],[4,5],[4,5],[4,5],[4,5],[4,5]])
L = np.ones(M.shape[0])
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
               'accuracy': accuracy_score(L[test_index], pred)}
  return ret
 
results = run(RandomForestClassifier, data, clf_hyper={})


#testing area
clfTest = [LogisticRegression, {'C': 1.0, 'solver': 'lbfgs', 'tol' : 1e-3}]

clf = clfTest[0]
clf = clf(**clfTest[1])
clf.fit
  