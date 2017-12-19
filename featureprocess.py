import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import csv

samples = np.loadtxt(open("./data/train.csv","rb"),delimiter=",")
X = samples[:,2:]
y = samples[:,0]
# Create the RFE object and rank each pixel
def rfe4feature():
  svc = SVC(kernel="linear", C=1)
  rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
  rfe.fit(X, y)
  ranking = rfe.ranking_.reshape([150,75])
# Plot pixel ranking
  plt.matshow(ranking, cmap=plt.cm.Blues)
  plt.colorbar()
  plt.title("Ranking of pixels with RFE")
  plt.show()

def linearsvc4feature():
  lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X,y)
  model = SelectFromModel(lsvc, prefit=True)
  X_new = model.transform(X)
  print X.shape, X_new.shape

  csvfile = open('./data/train_fs.csv', 'w')
  writer = csv.writer(csvfile)
  for i in range(len(X_new)):
    con = np.hstack((samples[i,0:2],X_new[i,:]))
    writer.writerow(con)
  csvfile.close()

linearsvc4feature()
