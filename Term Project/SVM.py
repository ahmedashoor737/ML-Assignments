import pandas as pd
import numpy as np
from sklearn.svm import SVC
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import normalize
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


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


#File names
fv = './facies_vectors.csv'
test_data = './test_data_nofacies.csv'

#Reading CSVs
fv_df = pd.read_csv(fv)
test_data_df = pd.read_csv(test_data)

#Dealing with NaNs
fv_df = fv_df.fillna(0)
test_data_df = test_data_df.fillna(0)






#DataFrame Exploration
# print fv_df
# print 82 * '_'
# print test_data_df


#Training & Validation Segment

X = fv_df[['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X = X.as_matrix()
# X = normalize(X, axis=0)

Y = fv_df['Facies']
Y = Y.as_matrix()

svc = SVC(decision_function_shape='ovr', kernel='rbf', C=2)

percentage = .9
reg_acc = use_SVC_reg(X,Y,percentage,svc)
print 'Accuracy at {0:}% Seperation: {1:.2f}\n'.format(percentage*100, reg_acc)
folds = 10
print 'Using {}-Folds'.format(folds)
kfold_acc, kfold_ind = use_kfolds(X,Y,folds,svc)
print 'KFolds Accuracies: {0:},\nhighest: {1:.2f} - {2:}'.format(kfold_acc,kfold_acc[np.argmax(kfold_acc)], kfold_ind[np.argmax(kfold_acc)])
print 82 * '_'


print 'NO PE'
X = fv_df[['GR','ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X = X.as_matrix()
# X = normalize(X, axis=0)

Y = fv_df['Facies']
Y = Y.as_matrix()

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