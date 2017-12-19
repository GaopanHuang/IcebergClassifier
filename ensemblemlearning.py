import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

train_num = 1024
samples = np.loadtxt(open("./data/train.csv","rb"),delimiter=",",skiprows=0)
y_test = samples[train_num:, 0]


rlt_bagging = np.loadtxt(open("./results/baggingpro.csv","rb"),delimiter=",",skiprows=2)
rlt_rf = np.loadtxt(open("./results/rfpro.csv","rb"),delimiter=",",skiprows=2)
#rlt_svm = np.loadtxt(open("./results/svmpro.csv","rb"),delimiter=",",skiprows=2)
rlt_adaboost = np.loadtxt(open("./results/adaboostpro.csv","rb"),delimiter=",",skiprows=2)
#rlt = (rlt_bagging[:,1:]+rlt_rf[:,1:]+rlt_svm[:,1:]+rlt_adaboost[:,1:])/4
rlt = (rlt_bagging[:,1:]+rlt_rf[:,1:]+rlt_adaboost[:,1:])/3

onehot = OneHotEncoder(n_values=2, sparse=False)
y_test_true = onehot.fit_transform(y_test.reshape(-1,1))
logloss = log_loss(y_test_true,rlt)
print ('logloss:%f' % logloss)

y_predict = np.argmax(rlt,axis = 1)
acc = np.mean(np.equal(y_test,y_predict))
print acc

csvfile = open('./results/ensemblepro.csv', 'w')
writer = csv.writer(csvfile)
writer.writerow([logloss])
for i in range(len(rlt)):
  con = np.hstack((i,rlt[i,:]))
  writer.writerow(con)
csvfile.close()