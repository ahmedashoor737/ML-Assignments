import pandas as pd
from sklearn.preprocessing import normalize

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
	

def get(without_PE=False, normalize_X=False, fill_na=False, balance_data=False, verbose=False):
	fv_df, test_data_df = read_dataframes()

	if fill_na:
		fv_df = fv_df.fillna(0)
		test_data_df = test_data_df.fillna(0)

	if balance_data:
		fv_df = balanced(fv_df, verbose)

	features = ['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']
	if without_PE:
		features = ['GR','ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']

	X_df = fv_df[features]
	X = X_df.as_matrix()
	if normalize_X:
		X = normalize(X, axis=0)
	
	the_class = 'Facies'

	Y_df = fv_df[the_class]
	Y = Y_df.as_matrix()

	X_test_df = test_data_df[features]
	X_test = X_test_df.as_matrix()

	return X, Y, X_test

def balanced(fv_df, verbose=False):
	if verbose:
		# before balance
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

	balanced_fv_df = pd.concat(d)

	if verbose:
		# after balance
		print 'Facies 1: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==1]), len(balanced_fv_df[balanced_fv_df['Facies']==1])*100/len(balanced_fv_df))
		print 'Facies 2: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==2]), len(balanced_fv_df[balanced_fv_df['Facies']==2])*100/len(balanced_fv_df))
		print 'Facies 3: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==3]), len(balanced_fv_df[balanced_fv_df['Facies']==3])*100/len(balanced_fv_df))
		print 'Facies 4: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==4]), len(balanced_fv_df[balanced_fv_df['Facies']==4])*100/len(balanced_fv_df))
		print 'Facies 5: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==5]), len(balanced_fv_df[balanced_fv_df['Facies']==5])*100/len(balanced_fv_df))
		print 'Facies 6: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==6]), len(balanced_fv_df[balanced_fv_df['Facies']==6])*100/len(balanced_fv_df))
		print 'Facies 7: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==7]), len(balanced_fv_df[balanced_fv_df['Facies']==7])*100/len(balanced_fv_df))
		print 'Facies 8: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==8]), len(balanced_fv_df[balanced_fv_df['Facies']==8])*100/len(balanced_fv_df))
		print 'Facies 9: {} - {}%'.format(len(balanced_fv_df[balanced_fv_df['Facies']==9]), len(balanced_fv_df[balanced_fv_df['Facies']==9])*100/len(balanced_fv_df))

		print '\n\n'
		print 82 * '_'

	return balanced_fv_df
