from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn import metrics
from our_metrics import print_performance
import data

#Functions Segment




#End of Functions Segment

X, Y, X_test = data.get(normalize_X=True, fill_na=True, balance_data=True,
    show_class_distribution=True, verbose=True)

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

print_performance('MLP balanced PE', Y_vald, y_hat)