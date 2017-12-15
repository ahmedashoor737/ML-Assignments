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
from sklearn.tree import DecisionTreeClassifier

def find_best_estimator(X, y):
    cachedir = mkdtemp()

    # Classifier and preprocessing
    pipe = Pipeline([
        ('impute', Imputer()),
        ('reduce_dim', PCA()),
        ('clf', DecisionTreeClassifier())],
        memory=cachedir)

    # Parameter and preprocessing tuning space
    param_grid = [
        {
            'impute__strategy': ['mean', 'median', 'most_frequent'],
            'reduce_dim': [None, Normalizer(), PCA(4), PCA(5), PCA(6)],
            'clf__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'clf__class_weight': [None, 'balanced']
        }
    ]
    if (X.shape[1] == 7):
        param_grid[0]['reduce_dim'].append(PCA(7))

    # Search using relaxed metric
    scoring = {'relaxed': make_scorer(relaxed_accuracy), 'accuracy': make_scorer(accuracy_score)}
    grid = GridSearchCV(pipe, param_grid, scoring=scoring['relaxed'])

    # start search
    grid.fit(X, y)

    rmtree(cachedir)

    return grid

# With PE
name = 'DT PE'
X, y, X_no_labels = data.get()
grid = find_best_estimator(X, y)
report_performance(name, X, y, grid=grid)

# Without PE
name = 'DT No PE'
X, y, X_no_labels = data.get(without_PE=True)
grid = find_best_estimator(X, y)
report_performance(name, X, y, grid=grid)