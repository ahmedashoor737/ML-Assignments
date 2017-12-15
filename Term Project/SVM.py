import numpy as np
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score
from our_metrics import relaxed_accuracy, print_performance
from sklearn.model_selection import train_test_split, cross_validate
import data



X, Y, X_test = data.get(fill_na_strategy = 'mean', balance_data = True, verbose=True)

scal = StandardScaler()
scal.fit(X)
X = scal.transform(X)
percentage = .9
svc = SVC(decision_function_shape='ovr', kernel='rbf', C=3)



folds = 10
print 'Using {}-Folds'.format(folds)
scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
scores = cross_validate(svc, X, Y, scoring=scoring, cv=folds, return_train_score=True)
print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))

print 82 * '_'


print 'NO PE'

X, Y, X_test = data.get(without_PE=True, fill_na_strategy = 'mean', balance_data = True, verbose=True)
svc = SVC(decision_function_shape='ovo', kernel='rbf')

scal.fit(X)
X = scal.transform(X)

percentage = .9

folds = 10
print 'Using {}-Folds'.format(folds)
scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
scores = cross_validate(svc, X, Y, scoring=scoring, cv=folds, return_train_score=True)
print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))
print 82 * '_'
