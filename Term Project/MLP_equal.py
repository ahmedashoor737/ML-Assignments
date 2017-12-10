import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import normalize


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

print 'Facies 1: {} - {}%'.format(len(fv_df[fv_df['Facies']==1]), len(fv_df[fv_df['Facies']==1])*100/len(fv_df))
print 'Facies 2: {} - {}%'.format(len(fv_df[fv_df['Facies']==2]), len(fv_df[fv_df['Facies']==2])*100/len(fv_df))
print 'Facies 3: {} - {}%'.format(len(fv_df[fv_df['Facies']==3]), len(fv_df[fv_df['Facies']==3])*100/len(fv_df))
print 'Facies 4: {} - {}%'.format(len(fv_df[fv_df['Facies']==4]), len(fv_df[fv_df['Facies']==4])*100/len(fv_df))
print 'Facies 5: {} - {}%'.format(len(fv_df[fv_df['Facies']==5]), len(fv_df[fv_df['Facies']==5])*100/len(fv_df))
print 'Facies 6: {} - {}%'.format(len(fv_df[fv_df['Facies']==6]), len(fv_df[fv_df['Facies']==6])*100/len(fv_df))
print 'Facies 7: {} - {}%'.format(len(fv_df[fv_df['Facies']==7]), len(fv_df[fv_df['Facies']==7])*100/len(fv_df))
print 'Facies 8: {} - {}%'.format(len(fv_df[fv_df['Facies']==8]), len(fv_df[fv_df['Facies']==8])*100/len(fv_df))
print 'Facies 9: {} - {}%'.format(len(fv_df[fv_df['Facies']==9]), len(fv_df[fv_df['Facies']==9])*100/len(fv_df))


print '\n\n'
print 82 * '_'

#Taking equal data based on minority class
fd_1 = fv_df[fv_df['Facies']==1]
fd_2 = fv_df[fv_df['Facies']==2]
fd_3 = fv_df[fv_df['Facies']==3]
fd_4 = fv_df[fv_df['Facies']==4]
fd_5 = fv_df[fv_df['Facies']==5]
fd_6 = fv_df[fv_df['Facies']==6]
fd_7 = fv_df[fv_df['Facies']==7]
fd_8 = fv_df[fv_df['Facies']==8]
fd_9 = fv_df[fv_df['Facies']==9]


d = [fd_1[:len(fd_7)],fd_2[:len(fd_7)],fd_3[:len(fd_7)],fd_4[:len(fd_7)],fd_5[:len(fd_7)],fd_6[:len(fd_7)],fd_7[:len(fd_7)],fd_8[:len(fd_7)],fd_9[:len(fd_7)]]

fv_df = pd.concat(d)

print 'Facies 1: {} - {}%'.format(len(fv_df[fv_df['Facies']==1]), len(fv_df[fv_df['Facies']==1])*100/len(fv_df))
print 'Facies 2: {} - {}%'.format(len(fv_df[fv_df['Facies']==2]), len(fv_df[fv_df['Facies']==2])*100/len(fv_df))
print 'Facies 3: {} - {}%'.format(len(fv_df[fv_df['Facies']==3]), len(fv_df[fv_df['Facies']==3])*100/len(fv_df))
print 'Facies 4: {} - {}%'.format(len(fv_df[fv_df['Facies']==4]), len(fv_df[fv_df['Facies']==4])*100/len(fv_df))
print 'Facies 5: {} - {}%'.format(len(fv_df[fv_df['Facies']==5]), len(fv_df[fv_df['Facies']==5])*100/len(fv_df))
print 'Facies 6: {} - {}%'.format(len(fv_df[fv_df['Facies']==6]), len(fv_df[fv_df['Facies']==6])*100/len(fv_df))
print 'Facies 7: {} - {}%'.format(len(fv_df[fv_df['Facies']==7]), len(fv_df[fv_df['Facies']==7])*100/len(fv_df))
print 'Facies 8: {} - {}%'.format(len(fv_df[fv_df['Facies']==8]), len(fv_df[fv_df['Facies']==8])*100/len(fv_df))
print 'Facies 9: {} - {}%'.format(len(fv_df[fv_df['Facies']==9]), len(fv_df[fv_df['Facies']==9])*100/len(fv_df))


print '\n\n'
print 82 * '_'


X_train_df = fv_df[['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X = X_train_df.as_matrix()
X = normalize(X, axis=0)
X_train = X[:int(len(X)*0.7)]
X_vald = X[int(len(X)*0.7):]
Y_train_df = fv_df['Facies']
Y = Y_train_df.as_matrix()
Y_train = Y[:int(len(Y)*0.7)]
Y_vald = Y[int(len(Y)*0.7):]

X_test_df = test_data_df[['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X_test = X_test_df.as_matrix()
# Y_test_df = test_data_df['Facies']
# Y_test = Y_test_df.as_matrix()

mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)

mlp.fit(X_train,Y_train)

y_hat = mlp.predict(X_vald)

accuracy = 0.0
count = 0.0
for i in range(0, len(y_hat)):
    if y_hat[i] == Y_vald[i]:
        count += 1

accuracy = count/len(y_hat)
print 'Accuracy: {0:.2f}'.format(accuracy)