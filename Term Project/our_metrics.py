import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate

# Consider it true if adjacent according to adjacency table
def relaxed_accuracy(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8])

	correct = 0.0
	total = 0.0
	for i in xrange(9):
		for j in xrange(9):
			samples_of_i_predicted_as_j = matrix[i,j]

			# to start from 1 for easy comparison with adjacency table
			ip = i + 1
			jp = j + 1

			is_correct = \
				(ip == jp) or \
				(ip == 1 and (jp == 2)) or \
				(ip == 2 and (jp == 1 or jp == 3)) or \
				(ip == 3 and (jp == 2)) or \
				(ip == 4 and (jp == 5)) or \
				(ip == 5 and (jp == 4 or jp == 6)) or \
				(ip == 6 and (jp == 5 or jp == 7)) or \
				(ip == 7 and (jp == 6 or jp == 8)) or \
				(ip == 8 and (jp == 6 or jp == 7 or jp == 9)) or \
				(ip == 9 and (jp == 7 or jp == 8))

			if is_correct:
				correct += samples_of_i_predicted_as_j

			total += samples_of_i_predicted_as_j

	return correct / total

# k: run 5 times by default
def report_performance(name, grid, X_train, y_train, X_test, y_test, k=5):
	clf = grid.best_estimator_

	# scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
	# scores = cross_validate(clf, X, y, scoring=scoring, cv=k, return_train_score=True)

	print name
	print '\n ', grid.best_params_, '\n'

	# import numpy as np
	print ' Accuracy -Train {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train)))
	print ' Relaxed  -Train {:.2f}'.format(relaxed_accuracy(y_train, clf.predict(X_train)))

	print ' Accuracy -Test {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test)))
	print ' Relaxed  -Test {:.2f}'.format(relaxed_accuracy(y_test, clf.predict(X_test)))
	# print ' Accuracy avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy']))
	# print ' Relaxed  avg train {:.2f} | test {:.2f}'.format(np.mean(scores['train_relaxed']), np.mean(scores['test_relaxed']))

def print_performance(classifier_name, y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	relaxed = relaxed_accuracy(y_true, y_pred)

	print classifier_name
	print ' accuracy:', accuracy
	print ' relaxed:', relaxed, '\n'

	return accuracy, relaxed

def print_multiple(name, k, clf, X, y, print_all=False):
	accuracies = []
	relaxed_scores = []
	for i in xrange(k):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		clf.fit(X_train, y_train)
		y_test_predict = clf.predict(X_test)

		if print_all:
			accuracy, relaxed = print_performance('{}. {}'.format(k, name), y_test, y_test_predict)
		else:
			accuracy = accuracy_score(y_test, y_test_predict)
			relaxed = relaxed_accuracy(y_test, y_test_predict)

		accuracies.append(accuracy)
		relaxed_scores.append(relaxed)

	print name
	print ' max / avg accuracy:', max(accuracies), np.mean(accuracies)
	print ' max / avg relaxed: ' , max(relaxed_scores), np.mean(relaxed_scores), '\n'

if __name__ == '__main__':
	print 'Testing relaxed_accuracy()'

	import numpy as np
	y_true = np.arange(9)+1
	y_pred = np.zeros(9)+1
	print ' true:', y_true
	print ' pred:', y_pred
	print ' Should be (2/9=0.222):', relaxed_accuracy(y_true, y_pred), '\n'

	y_true = np.arange(9)+1
	y_pred = np.arange(9)+1
	print ' true:', y_true
	print ' pred:', y_pred
	print ' Should be (1):', relaxed_accuracy(y_true, y_pred)