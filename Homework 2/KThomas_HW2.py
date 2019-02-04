import csv
import functools
import io
import numpy as np
import sys
import numpy.lib.recfunctions as rfn
import time
import pandas as pd



hw_file = '/home/kyle_thomas/Documents/For_Others/ME/SMU/machine_learning_class/Homework 2/claim.sample.csv'

with open(hw_file) as f:
    reader = csv.reader(f, delimiter=',')

### USING CSV READER
# https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting
c = csv.reader(open(hw_file, 'r'), delimiter=',')
c2 = list(c)
claims = np.array(c2)
claims = list(reader)

### USING np.loadtxt

claims0 = np.genfromtxt(hw_file, delimiter=',')
claims1 = np.recfromcsv(hw_file, delimiter=',', filling_values = np.nan, case_sensitive=True, deletechars='', replace_space=' ')





# with pandas the approach can be as simple as:
temp = pd.read_csv(hw_file)
claims = temp.values


### APPROACH FROM OFFICE HOURS
#Read the two first two lines of the file.
genfromtxt_old = np.genfromtxt
@functools.wraps(genfromtxt_old)
def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
  if isinstance(f, io.TextIOBase):
    if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
    isinstance(f.buffer.raw, io.FileIO):
      # Best case: get underlying FileIO stream (binary!) and use that
      fb = f.buffer.raw
      # Reset cursor on the underlying object to match that on wrapper
      fb.seek(f.tell())
      result = genfromtxt_old(fb, *args, **kwargs)
      # Reset cursor on wrapper to match that of the underlying object
      f.seek(fb.tell())
    else:
      # Not very good but works: Put entire contents into BytesIO object,
      # otherwise same ideas as above
      old_cursor_pos = f.tell()
      fb = io.BytesIO(bytes(f.read(), encoding=encoding))
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(old_cursor_pos + fb.tell())
  else:
    result = genfromtxt_old(f, *args, **kwargs)
  return result

if sys.version_info >= (3,):
  np.genfromtxt = genfromtxt_py3_fixed




with open('data\claim.sample.csv', 'r') as f:
    print(f.readline())
    print(f.readline())

    
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
'''
typesCheck = [np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(float),
         np.dtype(object), np.dtype(float), np.dtype(float), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(float),
         np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(float),
         np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(object),
         np.dtype(object)]
'''

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


#      B. How much was paid for J-codes to providers for 'in network' claims?
j_codes['InOutOfNetwork'] #I,O, and " ". I will assume that in network is I and everything else is out of network
in_network_index = np.flatnonzero(np.core.defchararray.find(j_codes['InOutOfNetwork'], 'I'.encode() != -1))
sum_provider_payment = np.sum(j_codes["ProviderPaymentAmount"][in_network_index])
sum_subscriber_payment = np.sum(j_codes["SubscriberPaymentAmount"][in_network_index])
total_payments = sum_provider_payment + sum_subscriber_payment
# 2443969.2915400006
#      C. What are the top five J-codes based on the payment to providers?