import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

train_num = 1024
samples = np.loadtxt(open("./data/train.csv","rb"),delimiter=",",skiprows=0)

x_train = samples[0:train_num,2:]
y_train = samples[0:train_num, 0]
x_test = samples[train_num:,2:]
y_test = samples[train_num:, 0]


estimator_num = np.array([10, 40, 100, 300, 600, 1000])

print 'start'
for i in estimator_num:
  print i
  clf_bagging = BaggingClassifier(n_estimators=i, base_estimator=None, 
      max_samples=0.6, max_features=1.0, bootstrap=True, 
      bootstrap_features=False, oob_score=False, warm_start=False, 
      n_jobs=1, random_state=None, verbose=0)    
  clf_rf = RandomForestClassifier(n_estimators=i, criterion='gini', 
      max_depth=None, min_samples_split=2, min_samples_leaf=1, 
      min_weight_fraction_leaf=0.0, max_features='auto', 
      max_leaf_nodes=None, min_impurity_decrease=0.0, 
      min_impurity_split=None, bootstrap=True, oob_score=False, 
      n_jobs=1, random_state=None, verbose=0, warm_start=False, 
      class_weight=None)
  clf_adaboost = AdaBoostClassifier(n_estimators=i,
            base_estimator=DecisionTreeClassifier(max_depth=1),algorithm="SAMME",
            learning_rate=1.0, random_state=None)
  clf_etc = ExtraTreesClassifier(n_estimators=i, criterion='gini', 
      max_depth=None, min_samples_split=2, min_samples_leaf=1, 
      min_weight_fraction_leaf=0.0, max_features='auto', 
      max_leaf_nodes=None, min_impurity_decrease=0.0, 
      min_impurity_split=None, bootstrap=False, oob_score=False, 
      n_jobs=1, random_state=None, verbose=0, warm_start=False, 
      class_weight=None)
  clf_gbc = GradientBoostingClassifier(n_estimators=i, loss='deviance',
      learning_rate=0.1, subsample=1.0, criterion='friedman_mse', 
      min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
      max_depth=3, min_impurity_decrease=0.0, 
      min_impurity_split=None, 
      init=None, random_state=None, max_features=None, verbose=0, 
      max_leaf_nodes=None, warm_start=False, presort='auto')
  #clf_svm = svm.NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto', 
  #    coef0=0.0, shrinking=True, probability=False, tol=0.001, 
  #    cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
  #    decision_function_shape='ovr', random_state=None)

  for clf, fn, fn2 in zip([clf_bagging, clf_rf, clf_adaboost, clf_etc,clf_gbc], 
      ['bagging', 'rf', 'adaboost', 'etc','gbc'],
      ['baggingpro','rfpro','adaboostpro','etcpro','gbcpro']):
    print fn
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    acc = np.mean(np.equal(y_test, y_predict))
    print ('acc:%f' % acc)
    y_prepro = clf.predict_proba(x_test)
    onehot = OneHotEncoder(n_values=2, sparse=False)
    y_test_true = onehot.fit_transform(y_test.reshape(-1,1))
    logloss = log_loss(y_test_true,y_prepro)
    print ('logloss:%f' % logloss)
    
    csvfile = open(('./results/%s%d.csv' % (fn, i)), 'w')
    writer = csv.writer(csvfile)
    writer.writerow([acc])
    writer.writerow([logloss])
    for j in range(len(y_predict)):
      con = np.hstack((j,y_test[j],y_predict[j]))
      writer.writerow(con)
    csvfile.close()
    
    csvfile = open(('./results/%s%d.csv' % (fn2, i)), 'w')
    writer = csv.writer(csvfile)
    for j in range(len(y_prepro)):
      con = np.hstack((j,y_prepro[j,:]))
      writer.writerow(con)
    csvfile.close()
