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
  {'clf': [SVC], 'C': [1.1, 0.5], 'kernel': ['rbf'], 'tol': [1e-4] }
]


def unpackHypers(kwargs):
  #borrowed from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

# testInput =  [LogisticRegression, {'C': [0.5, 1.0, 1.5], 'solver': ['lbfgs', 'liblinear'], 'tol' : [1e-4, 1e-3, 1e-2]}]
def DeathToGridSearch(searchList):

  master_results = {}
  best_model = None

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
            error_msg = f"Classifier {classifier} failed with the following error:\n{e}\nResuming Search."
            print(error_msg)
        
          cls_name = str(classifier).split(".")[-1] #this is where the actual classifer is in the class
          cls_name = cls_name.replace("'>","")

          master_results[cls_name] = res


  # find the best results
  def find_best_model():
    temp_best_score = 0
    temp_best_model = None

    for k in master_results.keys():
      this_clf = master_results[k]

      for i in range(len(this_clf)):
        if this_clf[i]['accuracy'] > temp_best_score:
          temp_best_score = this_clf[i]['accuracy']
          temp_best_model = this_clf[i]
    
    return temp_best_model

  best_model = find_best_model()
    
  return master_results, best_model

my_results, my_best_model = DeathToGridSearch(testInput)


def plotResults(results_dict, save_path = '/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/model_results/'):
  """ Takes a results dictionary from run(). Generates plots to help evaluation. \n
  Plots that will be included: \n
  1. Accuracy Plots, total and by group. \n
  2. ROC Curves \n
  More as time permits"""

  if save_path[-1] != '/':
    save_path += '/'

  # master variables
  all_accs = []

  def accuracy_plot_by_model():
    for k in results_dict.keys():
      clf_accs = []
      this_clf = results_dict[k]

      for i in range(len(this_clf)):
        this_acc = this_clf[i]['accuracy']
        clf_accs.append(this_acc)
        all_accs.append(this_acc)
    
      # Plot the accuracies
      plt.plot(clf_accs)
      plt.title(f'{k} Accuracy Scores')
      plt.xlabel("Fold")
      plt.ylabel("Accuracy")
      # plt.show()
      plt.savefig(f'{save_path}{k}_Accuracy_Scores.png')
      plt.close()

  def accuracy_plot_all():
    clfs = []
    scores = []
    for k in results_dict.keys():
      this_clf = results_dict[k]
      max_acc = max(float(d['accuracy']) for d in this_clf.values())
      clfs.append(str(k))
      scores.append(max_acc)

    y_pos = np.arange(len(clfs))
    plt.bar(y_pos, scores, align='center')
    plt.xticks(y_pos, clfs)
    plt.ylabel("Max Accuracy")
    plt.title("Max Accuracy by Classifier")
    # plt.show()
    plt.savefig(f'{save_path}Best_Accuracy_Score_By_CLF.png')
    
  accuracy_plot_by_model()
  accuracy_plot_all()
  

plotResults(my_results)






  # def model_scatter(results_dict)
  #   truth_data = M[results[0]['test_index']]
  #   truth_labels = L[results[0]['test_index']]
  #   pred_labels = results[0]['predictions']
  #   colors = ['b', 'g', 'r', 'c', 'm', 'y']


  #   # If there are 2 dimensions the we can do 2D Scatter Plots.
  #   if truth_data.ndim <= 2:
  #     x_vals = truth_data[:,0]
  #     y_vals = truth_data[:,1]
  #     titles = ["Truth", "Prediction"]
  #     labels= [truth_labels, pred_labels]

  #     for title, label in zip(titles, labels):
  #       plt.scatter(x_vals, y_vals, c=label, label=label)
  #       plt.legend()
  #       plt.title(title)
  #       plt.show()

    
plotResults(results)