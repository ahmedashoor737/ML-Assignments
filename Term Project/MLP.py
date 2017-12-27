import numpy as np
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn.metrics import accuracy_score, make_scorer
from our_metrics import relaxed_accuracy, print_performance, report_performance
from sklearn.preprocessing import Imputer, RobustScaler
from sklearn.model_selection import train_test_split, cross_validate
import data




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


mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)#max_iter=max_iter
# scal = StandardScaler()
# scal.fit(X_train)
# X_train = scal.transform(X_train)
transformers = [
		('impute', Imputer(strategy = 'mean')),
		('reduce_dim', RobustScaler()),
		('clf', mlp)]
grid_params = {
			}
name = 'MLP PE'
X, Y = data.get()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=None)
grid = data.find_best_estimator(X_train, Y_train, transformers, grid_params)
report_performance(name, grid, X_train, Y_train, X_test, Y_test)

# Without PE
name = 'MLP No PE'
X, Y = data.get(without_PE=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=None)
grid = data.find_best_estimator(X_train, Y_train, transformers, grid_params)
report_performance(name, grid, X_train, Y_train, X_test, Y_test)
# percentage = .9
# print 'PE'
# mlp.fit(X_train, Y_train)

# folds = 10
# print 'Using {}-Folds'.format(folds)
# scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
# scores = cross_validate(mlp, X_train, Y_train, scoring=scoring, cv=folds, return_train_score=True)
# print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
# print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))
# test_pred = mlp.predict(X_test)
# print 'Accuracy testing {:.2f}'.format(accuracy_score(test_pred, Y_test))
# print 'Relaxed testing {:.2f}'.format(relaxed_accuracy(test_pred, Y_test))
# print 82 * '_'
# print 'No PE'

# X, Y, X_test = data.get(without_PE=True, verbose=True, fill_na_strategy = 'mean', balance_data = True)

# scal.fit(X)
# X = scal.transform(X)
# percentage = .9
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06, random_state=None)
# mlp.fit(X_train, Y_train)

# folds = 10
# print 'Using {}-Folds'.format(folds)
# scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
# scores = cross_validate(mlp, X, Y, scoring=scoring, cv=folds, return_train_score=True)
# print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
# print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))
# print 82 * '_'