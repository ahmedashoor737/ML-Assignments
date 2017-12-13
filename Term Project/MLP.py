from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn import metrics
import data

#Functions Segment




#End of Functions Segment

X, Y, X_test = data.get(fill_na=True, normalize_X=True, verbose=True)


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

X_train = X[:int(len(X)*0.7)]
X_vald = X[int(len(X)*0.7):]
Y_train = Y[:int(len(Y)*0.7)]
Y_vald = Y[int(len(Y)*0.7):]

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
