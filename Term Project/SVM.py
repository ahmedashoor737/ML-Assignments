import numpy as np
from sklearn.svm import SVC
from collections import defaultdict
from sklearn import metrics
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from our_metrics import relaxed_accuracy, print_performance
import data

#Functions Segment
def use_kfolds(X,Y,n_splits,svm):
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    accuracy = []
    relaxed = []
    indices = []
    for train_in, test_in in kf.split(X):
        X_train = X[train_in]
        Y_train = Y[train_in]
        X_vald = X[test_in]
        Y_vald = Y[test_in]
        svm.fit(X_train, Y_train)
        y_hat = svm.predict(X_vald)
        accuracy.append(accuracy_score(Y_vald, y_hat))
        relaxed.append(relaxed_accuracy(Y_vald, y_hat))
        indices.append([[[train_in[0],train_in[-1]],[test_in[0],test_in[-1]]]])
    return accuracy,relaxed,indices

def use_SVC_reg(X,Y,percentage,svc):
    X_train = X[:int(len(X)*percentage)]
    X_vald = X[int(len(X)*percentage):]
    Y_train = Y[:int(len(Y)*percentage)]
    Y_vald = Y[int(len(Y)*percentage):]
    svc.fit(X_train,Y_train)
    y_hat = svc.predict(X_vald)
    return Y_vald, y_hat

def get_class_weights(df_Y):
    labels = np.unique(df_Y)
    class_count = []
    for i in range(0,len(labels)):
        class_count.append(len(df_Y[df_Y == labels[i]]))

    weights = {}
    for i in range(0, len(class_count)):
        weights[labels[i]] = len(df_Y) / (len(labels) * class_count[i])
    
    return weights


#End of Functions Segment

X, Y, X_test = data.get(fill_na=True, verbose=True) #, normalize_X=True)

#Training & Validation Segment

svc = SVC(decision_function_shape='ovr', kernel='rbf', C=2)

percentage = .9
Y_vald, y_hat = use_SVC_reg(X,Y,percentage,svc)
print_performance('SVC Accuracy at {0:}% Seperation PE'.format(percentage*100), Y_vald, y_hat)

folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_relaxed,kfold_ind = use_kfolds(X,Y,folds,svc)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)], np.mean(kfold_acc))
print 'KFolds Relaxed: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_relaxed,kfold_relaxed[np.argmax(kfold_relaxed)], kfold_ind[np.argmax(kfold_relaxed)], np.mean(kfold_relaxed))
print 82 * '_'


print 'NO PE'

X, Y, X_test = data.get(without_PE=True, fill_na=True, verbose=True) #, normalize_X=True)

svc = SVC(decision_function_shape='ovo', kernel='rbf')



percentage = .9
Y_vald, y_hat = use_SVC_reg(X,Y,percentage,svc)
print_performance('SVC Accuracy at {0:}% Seperation No PE'.format(percentage*100), Y_vald, y_hat)

folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_relaxed,kfold_ind = use_kfolds(X,Y,folds,svc)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)], np.mean(kfold_acc))
print 'KFolds Relaxed: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_relaxed,kfold_relaxed[np.argmax(kfold_relaxed)], kfold_ind[np.argmax(kfold_relaxed)], np.mean(kfold_relaxed))
print 82 * '_'


#Testing Segment