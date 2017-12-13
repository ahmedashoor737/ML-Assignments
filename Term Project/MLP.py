import numpy as np
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import data

#Functions Segment
def mlp_split(X,Y,percentage,mlp):
    x_train = X[:int(len(X)*percentage)]
    y_train = Y[:int(len(X)*percentage)]
    x_test = X[int(len(X)*percentage):]
    y_test = Y[int(len(X)*percentage):]

    X_train = np.array(x_train)
    Y_train = np.array(y_train)

    X_test = np.array(x_test)
    Y_test = np.array(y_test)

    mlp.fit(X_train, Y_train)
    Y_hat = mlp.predict(X_test)
    return accuracy_score(Y_test, Y_hat)

def mlp_use_kfolds(X,Y,n_splits,mlp):
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    accuracy = []
    indices = []
    for train_in, test_in in kf.split(X):
        X_train = X[train_in]
        Y_train = Y[train_in]
        X_vald = X[test_in]
        Y_vald = Y[test_in]
        mlp.fit(X_train, Y_train)
        y_hat = mlp.predict(X_vald)
        accuracy.append(accuracy_score(Y_vald, y_hat))
        indices.append([[[train_in[0],train_in[-1]],[test_in[0],test_in[-1]]]])
    return accuracy,indices



#End of Functions Segment

X, Y, X_test = data.get(normalize_X=True, fill_na=True, verbose=True)


#MLP PARAMS
hidden_layer_sizes=(4,25)
activation='tanh'
solver='adam'
alpha=0.0001
batch_size='auto'
learning_rate='constant'
learning_rate_init=0.001
power_t=0.5
max_iter=200000
shuffle=True
random_state=None
tol=0.0001
verbose=False
warm_start=False
momentum=0.9
nesterovs_momentum=True
early_stopping=False
validation_fraction=0.1
beta_1=0.9
beta_2=0.999
epsilon=1e-08

mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)

print 'PE'
percentage = .9
reg_acc = mlp_split(X,Y,percentage,mlp)
print '\n\nAccuracy at {0:}% Seperation: {1:.2f}\n'.format(percentage*100, reg_acc)
folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_ind = mlp_use_kfolds(X, Y, folds, mlp)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)])
print 82 * '_'

print 'No PE'

X, Y, X_test = data.get(without_PE=True, normalize_X=True, fill_na=True, verbose=True)

percentage = .9
reg_acc = mlp_split(X,Y,percentage,mlp)
print '\n\nAccuracy at {0:}% Seperation: {1:.2f}\n'.format(percentage*100, reg_acc)
folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_ind = mlp_use_kfolds(X, Y, folds, mlp)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)])
print 82 * '_'