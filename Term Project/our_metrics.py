from sklearn.metrics import confusion_matrix

# Consider it true if adjacent according to adjacency table
def relaxed_accuracy(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)

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