import numpy as np
import scipy.io as sio
import re
import nltk
from svm import *
from svmutil import *
## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

def tolist(a):
  try:
    return list(tolist(i) for i in a)
  except TypeError:
    return a

def readFile(file):
  with open(file) as f:
    file_contents=f.read()
  return file_contents  

def processEmail(email_contents):
  word_indices=list()
  vocabList=dict()
  with open('vocab.txt') as f:
    for line in f.readlines():
      vocabList[int(line.split()[0])]=line.split()[1]
  email_contents=email_contents.lower()
  # Strip all HTML
  # Looks for any expression that starts with < and ends with > and replace
  # and does not have any < or > in the tag it with a space
  email_contents=re.sub(r'<[^,.]+>',' ',email_contents)

  # Handle Numbers
  # Look for one or more characters between 0-9
  email_contents = re.sub(r'[0-9]+', 'number',email_contents)

  # Handle URLS
  # Look for strings starting with http:// or https://
  email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr',email_contents)

  # Handle Email Addresses
  # Look for strings with @ in the middle
  email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr',email_contents)

  # Handle $ sign
  email_contents = re.sub(r'[$]+', 'dollar',email_contents)
  print('=====Processed Email=====')
  search=re.search(r'\b\w+\b',email_contents)
  while search:
    search=re.search(r'\b\w+\b',email_contents)
    if search:
      str=search.group()
      email_contents=re.sub('.*?(?<='+str+')','',email_contents,1)
      str=re.sub('[^a-zA-Z0-9]','',str)
      stemmer=nltk.PorterStemmer()
      str=stemmer.stem(str)
      print(str,end=' ')
    else:
      print('\nNo more words in Email content')
      break
    for (i,item) in vocabList.items():
      if item==str:
        word_indices.append(i)
  return word_indices

def wb(model):
  alpha=np.array(model.get_sv_coef())
  sv=np.zeros([len(model.get_SV()),1899])
  for i,dic in enumerate(model.get_SV()):
    for (j,val) in dic.items():
      sv[i,j-1]=dic[j]    
  w=sv.T.dot(alpha)
  b=-model.rho[0]
  return w,b

print('\nPreprocessing sample email (emailSample1.txt)')

# Extract Features
file_contents = readFile('emailSample1.txt')
word_indices  = processEmail(file_contents)
features=list([0]*1899)
for i in word_indices:
  features [i]=1

print('Length of feature vector:',len(features))
print('Number of non-zero entries:',sum(features))

print('Training Linear SVM (Spam Classification)')
data=sio.loadmat('spamTrain.mat')
X=data['X']
y=data['y']
y=y.astype(float).flatten()
y[y==0]=-1
Xs,ys=tolist(X),tolist(y)
#====Training============
prob=svm_problem(ys,Xs)
param=svm_parameter('-t 0 -c 1')
m=svm_train(prob,param)
p_label,p_acc,p_vals=svm_predict(ys,Xs,m)
print('Training accuracy:',p_acc)
#=====Testing============
data=sio.loadmat('spamTest.mat')
Xtest=data['Xtest']
ytest=data['ytest']
ytest=ytest.astype(float).flatten()
ytest[ytest==0]=-1
Xst,yst=tolist(Xtest),tolist(ytest)
pt_label,pt_acc,pt_vals=svm_predict(yst,Xst,m)
print('Test accuracy:',pt_acc)
vocabList=dict()
with open('vocab.txt') as f:
  for line in f.readlines():
    vocabList[int(line.split()[0])]=line.split()[1]
print('Top predictors of spam:')
w,b=wb(m)
index=np.argsort(-w,axis=None)
ws=-np.sort(-w,axis=None)
for i in range(15):
  print(' %-15s (%f) ' %(vocabList[index[i]+1],ws[i]))

file_contents = readFile('spamSample2.txt');
word_indices  = processEmail(file_contents);
x=list([0]*1899)
for i in word_indices:
  x[i]=1
p_label,p_acc,p_val = svm_predict(list([1,]), list([x,]), m)

print('Processed spamSample2.txt\n\nSpam Classification: %d' %p_label[0]);
print('(1 indicates spam, -1 indicates not spam)\n');
