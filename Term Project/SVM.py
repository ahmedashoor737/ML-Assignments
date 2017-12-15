import numpy as np
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from our_metrics import relaxed_accuracy, print_performance
from sklearn.model_selection import train_test_split, cross_validate
import data



X, Y, X_test = data.get(fill_na=True, verbose=True) #, normalize_X=True)

scal = StandardScaler()
scal.fit(X)
X = scal.transform(X)
#Training & Validation Segment
percentage = .9
svc = SVC(decision_function_shape='ovr', kernel='rbf', C=3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06, random_state=None)
svc.fit(X_train, Y_train)
print_performance('SVC train Accuracy at {:.2f}% Seperation PE'.format(percentage*100), Y_train, svc.predict(X_train))
print_performance('SVC test Accuracy at {:.2f}% Seperation PE'.format(percentage*100), Y_test, svc.predict(X_test))

folds = 10
print 'Using {}-Folds'.format(folds)
scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
scores = cross_validate(svc, X, Y, scoring=scoring, cv=folds, return_train_score=True)
print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))

print 82 * '_'


print 'NO PE'

X, Y, X_test = data.get(without_PE=True, fill_na=True, verbose=True) #, normalize_X=True)

svc = SVC(decision_function_shape='ovo', kernel='rbf')

scal.fit(X)
X = scal.transform(X)

percentage = .9
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.06, random_state=None)
svc.fit(X_train, Y_train)
print_performance('SVC train Accuracy at {:.2f}% Seperation No PE'.format(percentage*100), Y_train, svc.predict(X_train))
print_performance('SVC test Accuracy at {:.2f}% Seperation No PE'.format(percentage*100), Y_test, svc.predict(X_test))

folds = 10
print 'Using {}-Folds'.format(folds)
scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
scores = cross_validate(svc, X, Y, scoring=scoring, cv=folds, return_train_score=True)
print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))
print 82 * '_'


#Testing Segment