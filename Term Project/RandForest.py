from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from collections import defaultdict
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from our_metrics import relaxed_accuracy, print_performance
import data

#Functions Segment
def rand_forest_split(X,Y,percentage,rand_forest):
    x_train = X[:int(len(X)*percentage)]
    y_train = Y[:int(len(X)*percentage)]
    x_test = X[int(len(X)*percentage):]
    y_test = Y[int(len(X)*percentage):]

    X_train = np.array(x_train)
    Y_train = np.array(y_train)

    X_test = np.array(x_test)
    Y_test = np.array(y_test)

    rand_forest.fit(X_train, Y_train)
    Y_hat = rand_forest.predict(X_test)
    return Y_test, Y_hat

def rand_forest_use_kfolds(X,Y,n_splits,rand_forest):
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    accuracy = []
    relaxed = []
    indices = []
    for train_in, test_in in kf.split(X):
        X_train = X[train_in]
        Y_train = Y[train_in]
        X_vald = X[test_in]
        Y_vald = Y[test_in]
        rand_forest.fit(X_train, Y_train)
        y_hat = rand_forest.predict(X_vald)
        accuracy.append(accuracy_score(Y_vald, y_hat))
        relaxed.append(relaxed_accuracy(Y_vald, y_hat))
        indices.append([[[train_in[0],train_in[-1]],[test_in[0],test_in[-1]]]])
    return accuracy,relaxed,indices



#End of Functions Segment

X, Y, X_test = data.get(normalize_X=True, fill_na=True, verbose=True)




rand_forest = RandomForestClassifier(max_depth=10, random_state=0)

print 'PE'
percentage = .9
Y_test, y_hat = rand_forest_split(X,Y,percentage,rand_forest)
print_performance('rand_forest Accuracy at {0:}% Seperation PE:'.format(percentage*100), Y_test, y_hat)

folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_relaxed, kfold_ind = rand_forest_use_kfolds(X, Y, folds, rand_forest)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)], np.mean(kfold_acc))
print 'KFolds Relaxed: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_relaxed,kfold_relaxed[np.argmax(kfold_relaxed)], kfold_ind[np.argmax(kfold_relaxed)], np.mean(kfold_relaxed))
print 82 * '_'

print 'No PE'

X, Y, X_test = data.get(without_PE=True, normalize_X=True, fill_na=True, verbose=True)

percentage = .9
Y_test, y_hat = rand_forest_split(X,Y,percentage,rand_forest)
print_performance('rand_forest Accuracy at {0:}% Seperation no PE:'.format(percentage*100), Y_test, y_hat)

folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_relaxed, kfold_ind = rand_forest_use_kfolds(X, Y, folds, rand_forest)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)], np.mean(kfold_acc))
print 'KFolds Relaxed: {0:},\nhighest: {1:.2f} - {2:},\nAverage: {3:.2f}'.format(kfold_relaxed,kfold_relaxed[np.argmax(kfold_relaxed)], kfold_ind[np.argmax(kfold_relaxed)], np.mean(kfold_relaxed))
print 82 * '_'