import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from shutil import rmtree
from tempfile import mkdtemp
from sklearn.metrics import make_scorer
from our_metrics import relaxed_accuracy
from sklearn.model_selection import GridSearchCV
#File names
fv = './facies_vectors.csv'
test_data = './test_data_nofacies.csv'

def read_dataframes():
	global fv, test_data

	fv_df = pd.read_csv(fv)
	test_data_df = pd.read_csv(test_data)
	
	#DataFrame Exploration
	# print fv_df
	# print 82 * '_'
	# print test_data_df

	return fv_df, test_data_df
	
'''
Parameters (mostly boolean)
 without_PE: exclude PE feature.
 normalize_X: normalizes X (maybe should return the norm).

 fill_na: Replace missing with 0.
 fill_na_strategy: Replace missing using the strategy. Can be 'mean', 'median', or 'most_frequent'.
 
 keep:
   If not None, will return object of class PCA.
   If int, number of features to reduce to.
   See PCA class for possible values.
 reduce_X_PCA: X will be reduced using PCA.reduce()
 reduce_X_test_PCA: X_test will be reduced too.

 balance_data: Make class distribution equal.
 show_class_distribution: Print class distribution.

Return
 X: features of facies_vectors.csv
 Y: labels of facies_vectors.csv
 X_test: features of test_data_nofacies.csv
 PCA_X: of class PCA which has method transform(X). Not returned if keep=None
'''
def find_best_estimator(X, y, tranformers, 	param_grid):
	cachedir = mkdtemp()

	# Classifier and preprocessing
	pipe = Pipeline(tranformers,
		memory=cachedir)

	# Search using relaxed metric
	grid = GridSearchCV(pipe, param_grid, scoring=make_scorer(relaxed_accuracy))

	# start search
	grid.fit(X, y)

	rmtree(cachedir)

	return grid
def get(without_PE=False, show_class_distribution=False):
	fv_df, test_data_df = read_dataframes()

	if show_class_distribution:
		print_distribution(fv_df)

	features = ['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
	if without_PE:
		features = ['GR','ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
	the_class = 'Facies'

	X_df = fv_df[features]
	# X_test_df = test_data_df[features]
	Y_df = fv_df[the_class]

	X = X_df.as_matrix()
	# X_test = X_test_df.as_matrix()
	Y = Y_df.as_matrix()

	return X, Y
def balanced(fv_df, verbose=False):
	if verbose:
		# before balance
		print_distribution(fv_df)

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

	balanced_fv_df = pd.concat(d)

	if verbose:
		# after balance
		print_distribution(balanced_fv_df)

	return balanced_fv_df

def print_distribution(fv_df):
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