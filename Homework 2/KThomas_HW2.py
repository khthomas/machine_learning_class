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





# hw_file = '/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/Homework 2/claim.sample.csv'
hw_file = '/home/kyle/Documents/thomaskh522@gmail.com/SMU/MachineLearning/machine_learning_class/Homework 2/claim.sample.csv'

# with open(hw_file) as f:
#     reader = csv.reader(f, delimiter=',')

# ### USING CSV READER
# # https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting
# c = csv.reader(open(hw_file, 'r'), delimiter=',')
# c2 = list(c)
# claims = np.array(c2)
# claims = list(reader)

# ### USING np.loadtxt

# claims0 = np.genfromtxt(hw_file, delimiter=',')
# claims1 = np.recfromcsv(hw_file, delimiter=',', filling_values = np.nan, case_sensitive=True, deletechars='', replace_space=' ')


### APPROACH FROM OFFICE HOURS
#Read the two first two lines of the file.
# genfromtxt_old = np.genfromtxt
# @functools.wraps(genfromtxt_old)
# def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
#   if isinstance(f, io.TextIOBase):
#     if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
#     isinstance(f.buffer.raw, io.FileIO):
#       # Best case: get underlying FileIO stream (binary!) and use that
#       fb = f.buffer.raw
#       # Reset cursor on the underlying object to match that on wrapper
#       fb.seek(f.tell())
#       result = genfromtxt_old(fb, *args, **kwargs)
#       # Reset cursor on wrapper to match that of the underlying object
#       f.seek(fb.tell())
#     else:
#       # Not very good but works: Put entire contents into BytesIO object,
#       # otherwise same ideas as above
#       old_cursor_pos = f.tell()
#       fb = io.BytesIO(bytes(f.read(), encoding=encoding))
#       result = genfromtxt_old(fb, *args, **kwargs)
#       f.seek(old_cursor_pos + fb.tell())
#   else:
#     result = genfromtxt_old(f, *args, **kwargs)
#   return result

# if sys.version_info >= (3,):
#   np.genfromtxt = genfromtxt_py3_fixed




# with open('data\claim.sample.csv', 'r') as f:
#     print(f.readline())
#     print(f.readline())

    
#Colunn names

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

#https://docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html
# '''
# typesCheck = [np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(float),
#          np.dtype(object), np.dtype(float), np.dtype(float), np.dtype(object),
#          np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(float),
#          np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(object),
#          np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(float),
#          np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(object),
#          np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(object),
#          np.dtype(object)]
# '''

#data types after using typesCheck instead of types in the below function
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']


CLAIMS = np.genfromtxt(hw_file, dtype=types, delimiter=',', names=True, usecols=list(range(29)))

#### START ANALYSIS
# 1. J-codes are procedure codes that start with the letter 'J'.

#      A. Find the number of claim lines that have J-codes.
j_codes_index = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], 'J'.encode())!=-1)
num_j_codes = len(j_codes_index)
# 51595
j_codes = CLAIMS[j_codes_index]
names = j_codes.dtype.names


#      B. How much was paid for J-codes to providers for 'in network' claims?
j_codes['InOutOfNetwork'] #I,O, and " ". I will assume that in network is I and everything else is out of network
in_network_index = np.flatnonzero(np.core.defchararray.find(j_codes['InOutOfNetwork'], 'I'.encode() != -1))
sum_provider_payment = np.sum(j_codes["ProviderPaymentAmount"][in_network_index])
sum_subscriber_payment = np.sum(j_codes["SubscriberPaymentAmount"][in_network_index])
total_payments = sum_provider_payment + sum_subscriber_payment
# 2443969.2915400006
#      C. What are the top five J-codes based on the payment to providers?

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


# 2. For the following exercises, determine the number of providers that were paid for at least one J-code. 
# Use the J-code claims for these providers to complete the following exercises.

# first find provider payment amounts greater than zero
provider_gtr_zero = np.nonzero(j_codes["ProviderPaymentAmount"])
provider_j_codes = j_codes[provider_gtr_zero]

number_of_providers_paid = len(provider_j_codes)
# 6313

#     A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) 
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
  ax.scatter(paid_dict[k], not_paid_dict[k], label=k)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.title("Paid vs Unpaid J Code Quantities by Provide ID")
plt.xlabel('Paid Quantity')
plt.ylabel('Unpaid Quantity')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

plt.scatter(paid_dict.values(), not_paid_dict.values())
plt.show()




#     B. What insights can you suggest from the graph?

#     C. Based on the graph, is the behavior of any of the providers concerning? Explain.


###### TESTING #####
# find j codes that have been paid, if the sum of the j codes is zero then no one was paid
j_codes_w_payments_gtr_zero = [y[0] for y in sorted_payments if y[1] > 0]
# encode the j codes since that is how they are stored in the master ndarray
j_codes_w_payments_gtr_zero = list(map(lambda x: x.encode(), j_codes_w_payments_gtr_zero))

# extract only these j_codes

index = list()
for jcode in j_codes_w_payments_gtr_zero:
  temp_index = np.flatnonzero(np.core.defchararray.find(j_codes['ProcedureCode'], jcode)!=-1)
  index.append(temp_index)

flattened_index = [val for sublist in index for val in sublist]


payment_index = np.flatnonzero(np.core.defchararray.find(j_codes['ProcedureCode'], ))
j_codes_index = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'], 'J'.encode())!=-1)