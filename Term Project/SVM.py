from sklearn.svm import SVC
import data
from sklearn.metrics import accuracy_score, make_scorer
from our_metrics import relaxed_accuracy, report_performance

# Caching for grid Search
from shutil import rmtree
from tempfile import mkdtemp
from sklearn.model_selection import GridSearchCV

# Pipeline, preprocessing, and classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


pipe = [
	('impute', Imputer(strategy = 'mean')),
	('reduce_dim', RobustScaler()),
	('clf', SVC(decision_function_shape='ovr', kernel='rbf', C = 50))]

# Parameter and preprocessing tuning space
# param_grid = [
# 	{
# 		'clf__C': [1, 2, 3, 5, 10, 50, 100], 50 is best
# 		'clf__class_weight': ['balanced']
# 	}
# ]
param_grid = {}
# With PE
name = 'LR PE'
X, Y = data.get()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06, random_state=None)
grid = data.find_best_estimator(X_train, Y_train, pipe, param_grid)
report_performance(name, grid, X_train, Y_train, X_test, Y_test)

# Without PE
name = 'LR No PE'
X, Y = data.get(without_PE=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06, random_state=None)
grid = data.find_best_estimator(X_train, Y_train, pipe, param_grid)
report_performance(name, grid, X_train, Y_train, X_test, Y_test)


# X, Y, X_test = data.get(fill_na_strategy = 'mean', balance_data = True, verbose=True)

# scal = StandardScaler()
# scal.fit(X)
# X = scal.transform(X)
# percentage = .9
# svc = SVC(decision_function_shape='ovr', kernel='rbf', C=3)



# folds = 10
# print 'Using {}-Folds'.format(folds)
# scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
# scores = cross_validate(svc, X, Y, scoring=scoring, cv=folds, return_train_score=True)
# print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
# print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))

# print 82 * '_'


# print 'NO PE'

# X, Y, X_test = data.get(without_PE=True, fill_na_strategy = 'mean', balance_data = True, verbose=True)
# svc = SVC(decision_function_shape='ovo', kernel='rbf')

# scal.fit(X)
# X = scal.transform(X)

# percentage = .9

# folds = 10
# print 'Using {}-Folds'.format(folds)
# scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
# scores = cross_validate(svc, X, Y, scoring=scoring, cv=folds, return_train_score=True)
# print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
# print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))
# print 82 * '_'
