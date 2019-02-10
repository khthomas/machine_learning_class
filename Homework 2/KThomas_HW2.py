import csv
import functools
import io
import numpy as np
import sys
import numpy.lib.recfunctions as rfn
import time
import operator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Model Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Import for death to gridsearch
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
from functools import reduce
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

# custom import
# import DeathToGridSearch
# from .. import DeathToGridSearch
# from . import Model_Search

# hw_file = '/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/Homework 2/claim.sample.csv'
hw_file = '/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/Homework_2/claim.sample.csv'

names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]

#data types after using typesCheck instead of types in the below function
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']


CLAIMS = np.genfromtxt(hw_file, dtype=types, delimiter=',', names=True, usecols=list(range(29)))


#################################################################################
# QUESTION 1
#################################################################################
#### START ANALYSIS
# 1. J-codes are procedure codes that start with the letter 'J'.

#      A. Find the number of claim lines that have J-codes.
# j_codes_index = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], 'J'.encode())!=-1) #consider changing this to 1
j_codes_index = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], 'J'.encode()) == 1) #consider changing this to 1

num_j_codes = len(j_codes_index)
# 51029
j_codes = CLAIMS[j_codes_index]
names = j_codes.dtype.names

#  B. How much was paid for J-codes to providers for 'in network' claims?
# Claim Charge amount?
j_codes['InOutOfNetwork'] #I,O, and " ". I will assume that in network is I and everything else is out of network
in_network_index = np.flatnonzero(np.core.defchararray.find(j_codes['InOutOfNetwork'], 'I'.encode() == 1))
sum_provider_payment = np.sum(j_codes["ProviderPaymentAmount"][in_network_index])
sum_subscriber_payment = np.sum(j_codes["SubscriberPaymentAmount"][in_network_index])
total_payments = sum_provider_payment + sum_subscriber_payment
# 2418603.6867250004
# providers only 2418429.572825



# C. What are the top five J-codes based on the payment to providers?

# first sort by payment provider
sorted_j_codes = np.sort(j_codes, order='ProviderPaymentAmount')

payment_dict = dict()

for row in range(len(sorted_j_codes)):
    p_code = sorted_j_codes['ProcedureCode'][row].decode()
    value = sorted_j_codes['ProviderPaymentAmount'][row]
    if p_code in payment_dict.keys():
        payment_dict[p_code].append(value)
    if not p_code in payment_dict.keys():
        payment_dict[p_code] = []
        payment_dict[p_code].append(value)

for key in payment_dict.keys():
    payment_dict[key] = functools.reduce((lambda x, y: x + y), payment_dict[key])

sorted_payments = sorted(payment_dict.items(), key = operator.itemgetter(1), reverse = True)

# TOP 5:
sorted_payments[:5]
# [('"J1745"', 434232.08058999985), ('"J0180"', 299776.560765), ('"J9310"', 168630.87357999998), ('"J3490"', 90249.91245000002), ('"J1644"', 81909.39601500018)]

#################################################################################
# QUESTION 2
#################################################################################
# 2. For the following exercises, determine the number of providers that were paid for at least one J-code. 
# Use the J-code claims for these providers to complete the following exercises.

# first find provider payment amounts greater than zero
provider_gtr_zero = np.nonzero(j_codes["ProviderPaymentAmount"])
provider_j_codes = j_codes[provider_gtr_zero]

number_of_providers_paid = len(provider_j_codes)
# 6068

#  A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) 
# for each provider versus the number of paid claims.

ref_index = list(range(len(j_codes)))
not_paid_index = [x for x in ref_index if x not in provider_gtr_zero[0]]
not_paid = j_codes[not_paid_index]

# what I need to do is make a data structure that has each provider (maybe as a key in a dict) then has the number paid and unpaid.
# Plot this. X = paid, Y = unpaid, should be one dot per provider

paid_dict = dict()
for provider in range(len(provider_j_codes)):
  this_provider = provider_j_codes[provider]["ProviderID"].decode()

  if this_provider in paid_dict.keys():
    paid_dict[this_provider] += 1
  if this_provider not in paid_dict.keys():
    paid_dict[this_provider] = 1


not_paid_dict = dict()

for provider in range(len(not_paid)):
  this_provider = not_paid[provider]["ProviderID"].decode()

  if this_provider in not_paid_dict.keys():
    not_paid_dict[this_provider] += 1
  if this_provider not in not_paid_dict.keys():
    not_paid_dict[this_provider] = 1

# based on this there are two providers that are not paid. This will make scattering difficult. Need to see which ones are missing
# and set vales to zero.

missing = [x for x in not_paid_dict.keys() if x not in paid_dict.keys()]
for k in missing:
  paid_dict[k] = 0

array_names = ['id', 'paid']
formats = ['S14', 'f8']
dtype = dict(names=array_names, formats=formats)

plot_array = np.array(list(paid_dict.items()), dtype = dtype)
plot_array2 = np.array(list(not_paid_dict.items()), dtype=dtype)
# make the scatter plot
# need to change data structure so that I can easily plot by grouping
# put legend outside graph https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for k in paid_dict.keys():
  xspace= np.linspace(0, 2000, 1000)
  ax.plot(xspace, xspace+0)
  ax.scatter(paid_dict[k], not_paid_dict[k], label=k)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.title("Paid vs Unpaid J Code Quantities by Provide ID")
plt.xlabel('Paid Quantity')
plt.ylabel('Unpaid Quantity')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

#     B. What insights can you suggest from the graph?
# Overall there are very few providers that tend to have a balance of paid vs unpaid j-codes.

#     C. Based on the graph, is the behavior of any of the providers concerning? Explain.
# Two of the providers have 0 payments, three providers have very few payments and lots of unpaid JCodes (FA0001387001, FA0001387002, FA0001389001).

###################################################################3

# 3. Consider all claim lines with a J-code.

#      A. What percentage of J-code claim lines were unpaid?
percent_unpaid = len(not_paid) / (len(provider_j_codes) + len(not_paid))
percent_unpaid
# 0.8776431824789224

#      B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
# Goal here to to predict when ‘Provider.Payment.Amount’ == 0. Will need to scrub the data a bit to make sure that
# it play's nicely with SKLean.
# Following some of Office Hours code
paid = provider_j_codes.copy()

new_dtype1 = np.dtype(not_paid.dtype.descr + [('IsUnpaid', '<i4')])
new_dtype2 = np.dtype(paid.dtype.descr + [('IsUnpaid', '<i4')])

Unpaid_Labeled = np.zeros(not_paid.shape, dtype=new_dtype1)
Paid_Labeled = np.zeros(paid.shape, dtype=new_dtype2)

# Copy data
for lab in not_paid.dtype.names:
  Unpaid_Labeled[lab] = not_paid[lab]

for lab in paid.dtype.names:
  Paid_Labeled[lab] = paid[lab]

# Assing Paid vs Not paid
Unpaid_Labeled['IsUnpaid'] = 1
Paid_Labeled['IsUnpaid'] = 0

# Combine the data
Data = np.concatenate((Unpaid_Labeled, Paid_Labeled), axis=0)

# Shuffle the data
np.random.shuffle(Data)

DataNames = Data.dtype.names

cat_features = ['V1', 'ProviderID','LineOfBusinessID','RevenueCode',
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'DiagnosisCode', 'DenialReasonCode',
                'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID',
                'AgreementID', 'ClaimType', ]

numeric_features = ['ClaimNumber', 'ClaimLineNumber', 'MemberID', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount', 'ProviderPaymentAmount',
                    'GroupIndex', 'SubscriberIndex', 'SubgroupIndex']

# Convert featuers to List
Mcat = np.array(Data[cat_features].tolist())
Mnum = np.array(Data[numeric_features].tolist())
L = np.array(Data['IsUnpaid'].tolist())

# First I want to see what variables are likely to be a predictor of IsUnpaid
# To start I will do a correaltion analysis of all numeric data types
def getHighCorrs(cutoff=0.5):
  vars_output = []
  corr_output = []
  for col in numeric_features:
    this_corr = np.corrcoef(L, Data[col])
    if abs(this_corr[0][1]) > cutoff:
      vars_output.append(col)
      corr_output.append(this_corr[0][1])
  return vars_output, corr_output

num_vars_to_use, the_corrs = getHighCorrs(0.1) # "high" is relative here.

# Next I want to do some feature reduction on the categroical variables
# To start I want to find variables that do not have a high cardnality
# High cardnality will wreck my system. Is this the best way to do it? 
# maybe not, but it is a start, if the model is good then I can keep it,
# if it sucks then I can re-run the analysis with the variables that did
# not make the cut.
def find_low_cards(num_unique = 20):
  output_cards = []
  cardnality = []

  for col in cat_features:
    card = len(np.unique(Data[col]))
    if card <= num_unique:
      output_cards.append(col)
      cardnality.append(card)
  
  return output_cards, cardnality

cat_vars_to_use, cardnality = find_low_cards()

# new featuers
Mcat2 = np.array(Data[cat_vars_to_use].tolist())
Mnum2 = np.array(Data[num_vars_to_use].tolist())
L = np.array(Data['IsUnpaid'].tolist())


le = LabelEncoder()

what_is_this = le.fit_transform(Mcat[1])

for i in range(len(cat_vars_to_use)):
   Mcat2[:,i] = le.fit_transform(Mcat2[:,i])

# Do one hot encoding
ohe = OneHotEncoder(sparse=False)
Mcat2 = ohe.fit_transform(Mcat2)

# check to make sure everything went smoothly
Mcat2.shape
Mnum2.shape

# Join the data sets and  make the data set to feed into the DeathToGridSearch Algo
M = np.concatenate((Mcat2, Mnum2), axis=1)
n_folds = 5
search_data = (M, L, n_folds)
M.shape

search_models = [
  {'clf': [LogisticRegression], 'C': [1.0], 'solver': ['lbfgs', 'liblinear'], 'tol' : [1e-3, 1e-4]},
  {'clf': [RandomForestClassifier], 'n_estimators': [10, 15, 20], 'criterion': ['gini', 'entropy']}
]


my_search = Model_Search(data=search_data)
my_results, my_best_model = my_search.DeathToGridSearch(search_models)

save_path = '/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/Homework_2/Model_Outputs'

my_search.plotResults(my_results, save_path=save_path)


# need to reshape to -1,1
# for model selection keep in mind runtime
# normalize continuous variables








# data_for_search = (Data, L, 5)
# test_models =  [
#     {'clf': [LogisticRegression], 'C': [1.0], 'solver': ['lbfgs'], 'tol' : [1e-3]},
#     {'clf': [RandomForestClassifier]},
#     {'clf': [SVC], 'C': [1.1, 0.5], 'kernel': ['rbf'], 'tol': [1e-4] }
#   ]

# dtg = Model_Search(data = data_for_search)

# models, best_model = dtg.DeathToGridSearch(test_models)


#   M = iris.data[:, :2] #taking the first two features, following example on sklearn website
#   L = iris.target
#   n_folds = 5
#   data = (M, L, n_folds)



#############################
# Death to Grid Search
#############################
class Model_Search:

  def __init__(self, data):
    """This class does custom searches of machine learning models. It is meant
    to be an alternative to GridSearch, or used to influence a smaller grid
    search space."""
    self.data = data

 
  def run(self, a_clf, data, clf_hyper={}):
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
                #  'AP': average_precision_score(L[test_index], pred),
                'predictions': pred}

    return ret

  def unpackHypers(self, kwargs):
    #borrowed from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
      keys = kwargs.keys()
      vals = kwargs.values()
      for instance in product(*vals):
          yield dict(zip(keys, instance))

# testInput =  [LogisticRegression, {'C': [0.5, 1.0, 1.5], 'solver': ['lbfgs', 'liblinear'], 'tol' : [1e-4, 1e-3, 1e-2]}]
  def DeathToGridSearch(self, searchList):

    master_results = {}
    best_model = None
    res = None

    for searchVar in searchList:
      allHypers = list(self.unpackHypers(searchVar))

      for search in allHypers:
            classifier = search.pop('clf', None) #if clf is a key pop it, return none otherwise
            hypers = search
            try:
            # some classifiers might fail, try to run it and if failure continue
              print('\n', classifier, hypers, '\n')
              res = self.run(classifier, self.data, hypers)
            except Exception as e:      
              error_msg = f"Classifier {classifier} failed with the following error:\n{e}\nResuming Search."
              res = None
              print(error_msg)
              continue
          
            cls_name = str(classifier).split(".")[-1] #this is where the actual classifer is in the class
            cls_name = cls_name.replace("'>","")

            while True:
              if cls_name in master_results.keys():
                cls_name += str(1)
              if not cls_name in master_results.keys():
                master_results[cls_name] = res
                break

    # find the best results
    def find_best_model():
      temp_best_score = 0
      temp_best_model = None

      for k in master_results.keys():
        this_clf = master_results[k]
        this_clf_accs = []

        for i in range(len(this_clf)):
            this_clf_accs.append(this_clf[i]['accuracy'])

        avg_acc  = reduce(lambda x,y: x+y, this_clf_accs) / len(this_clf_accs)

        if avg_acc > temp_best_score:
          temp_best_score = avg_acc
          temp_best_model = this_clf #the hypers should all be the same for each clf, keeping all info
            
      return temp_best_model

    best_model = find_best_model()
      
    return master_results, best_model



  def plotResults(self, results_dict, save_path = '/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/model_results/'):
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
        plt.ylabel("Accuracy/Recall")
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