import numpy as np
from sklearn.svm import SVC
from collections import defaultdict
from sklearn import metrics
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import data

#Functions Segment
def use_kfolds(X,Y,n_splits,svm):
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    accuracy = []
    indices = []
    for train_in, test_in in kf.split(X):
        X_train = X[train_in]
        Y_train = Y[train_in]
        X_vald = X[test_in]
        Y_vald = Y[test_in]
        svm.fit(X_train, Y_train)
        y_hat = svm.predict(X_vald)
        accuracy.append(accuracy_score(Y_vald, y_hat))
        indices.append([[[train_in[0],train_in[-1]],[test_in[0],test_in[-1]]]])
    return accuracy,indices

def use_SVC_reg(X,Y,percentage,svc):
    X_train = X[:int(len(X)*percentage)]
    X_vald = X[int(len(X)*percentage):]
    Y_train = Y[:int(len(Y)*percentage)]
    Y_vald = Y[int(len(Y)*percentage):]
    svc.fit(X_train,Y_train)
    y_hat = svc.predict(X_vald)
    return accuracy_score(Y_vald, y_hat)

#End of Functions Segment

X, Y, X_test = data.get(fill_na=True, verbose=True) #, normalize_X=True)

#Training & Validation Segment

svc = SVC(decision_function_shape='ovo', kernel='rbf')

percentage = .9
reg_acc = use_SVC_reg(X,Y,percentage,svc)
print 'Accuracy at {0:}% Seperation: {1:.2f}\n'.format(percentage*100, reg_acc)
folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_ind = use_kfolds(X,Y,folds,svc)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)])
print 82 * '_'


print 'NO PE'

X, Y, X_test = data.get(without_PE=True, fill_na=True, verbose=True) #, normalize_X=True)

svc = SVC(decision_function_shape='ovo', kernel='rbf')



percentage = .9
reg_acc = use_SVC_reg(X,Y,percentage,svc)
print 'Accuracy at {0:}% Seperation: {1:.2f}\n'.format(percentage*100, reg_acc)
folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_ind = use_kfolds(X,Y,10,svc)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)])
print 82 * '_'


#Testing Segment