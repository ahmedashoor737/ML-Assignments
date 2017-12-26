# Data, metrics
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

pipe = [
    ('impute', Imputer()),
    ('reduce_dim', RobustScaler()),
    ('clf', DecisionTreeClassifier())]

# Parameter and preprocessing tuning space
param_grid = [
    {
        'clf__max_depth': [None, 4, 5, 6, 7, 8, 9, 10],
        'clf__class_weight': ['balanced']
    }
]

# With PE
name = 'LR PE'
X, Y = data.get()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=None)
grid = data.find_best_estimator(X_train, Y_train, pipe, param_grid)
report_performance(name, grid, X_train, Y_train, X_test, Y_test)

# Without PE
name = 'LR No PE'
X, Y = data.get(without_PE=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=None)
grid = data.find_best_estimator(X_train, Y_train, pipe, param_grid)
report_performance(name, grid, X_train, Y_train, X_test, Y_test)
