import pandas as pd
import numpy as np
from sklearn.svm import SVC
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import normalize
import random


#Functions Segment




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





#DataFrame Exploration
# print fv_df
# print 82 * '_'
# print test_data_df




X_train_df = fv_df[['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
# X_train_df = fv_df[['GR','ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X = X_train_df.as_matrix()
# X = normalize(X, axis=0)
X_train = X[:int(len(X)*0.9)]
X_vald = X[int(len(X)*0.9):]

Y_train_df = fv_df['Facies']
Y = Y_train_df.as_matrix()
Y_train = Y[:int(len(Y)*0.9)]
Y_vald = Y[int(len(Y)*0.9):]

X_test_df = test_data_df[['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X_test = X_test_df.as_matrix()
# Y_test_df = test_data_df['Facies']
# Y_test = Y_test_df.as_matrix()



svc = SVC(decision_function_shape='ovo', kernel='rbf')

svc.fit(X_train,Y_train)

y_hat = svc.predict(X_vald)

accuracy = 0.0
count = 0.0
for i in range(0, len(y_hat)):
    if y_hat[i] == Y_vald[i]:
        count += 1
    # if y_hat[i] == Y[i]:
    #     count += 1

accuracy = count/len(y_hat)
print 'Accuracy: {0:.2f}'.format(accuracy)
print 82 * '_'
X_train_df = fv_df[['GR','ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
# X_train_df = fv_df[['GR','ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X = X_train_df.as_matrix()
# X = normalize(X, axis=0)
X_train = X[:int(len(X)*0.9)]
X_vald = X[int(len(X)*0.9):]

Y_train_df = fv_df['Facies']
Y = Y_train_df.as_matrix()
Y_train = Y[:int(len(Y)*0.9)]
Y_vald = Y[int(len(Y)*0.9):]

X_test_df = test_data_df[['GR','ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X_test = X_test_df.as_matrix()
# Y_test_df = test_data_df['Facies']
# Y_test = Y_test_df.as_matrix()



svc = SVC(decision_function_shape='ovo', kernel='rbf')

svc.fit(X_train,Y_train)

y_hat = svc.predict(X_vald)

accuracy = 0.0
count = 0.0
for i in range(0, len(y_hat)):
    if y_hat[i] == Y_vald[i]:
        count += 1
    # if y_hat[i] == Y[i]:
    #     count += 1

accuracy = count/len(y_hat)
print 'NO PE'
print 'Accuracy: {0:.2f}'.format(accuracy)