# Data, metrics
import data
from sklearn.metrics import accuracy_score, make_scorer
from our_metrics import relaxed_accuracy, report_performance

# Caching for grid Search
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.model_selection import GridSearchCV

# Pipeline, preprocessing, and classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def find_best_estimator(X, y):
	cachedir = mkdtemp()

	# Classifier and preprocessing
	pipe = Pipeline([
		('impute', Imputer()),
		('reduce_dim', PCA()),
		('clf', LogisticRegression())],
		memory=cachedir)

	# Parameter and preprocessing tuning space
	param_grid = [
		{
			'impute__strategy': ['mean', 'median', 'most_frequent'],
			'reduce_dim': [None, Normalizer(), PCA(4), PCA(5), PCA(6)],
			'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
			'clf__class_weight': [None, 'balanced']
		}
	]

	# Search using relaxed metric
	grid = GridSearchCV(pipe, param_grid, scoring=make_scorer(relaxed_accuracy))

	# start search
	grid.fit(X, y)

	rmtree(cachedir)

	return grid

# With PE
name = 'LR PE'
X, y, X_no_labels = data.get()
grid = find_best_estimator(X, y)
report_performance(name, X, y, grid=grid)

# Without PE
name = 'LR No PE'
X, y, X_no_labels = data.get(without_PE=True)
grid = find_best_estimator(X, y)
report_performance(name, X, y, grid=grid)
