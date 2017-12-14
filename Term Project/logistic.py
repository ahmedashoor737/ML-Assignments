from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import data

#Function Segment

#End of Functions Segment

# With PE
'''
fill
  fill_na
  fill_na_strategy: score a little bit higher than fill_na
    mean
    median
    most_frequent
normalize_X: Worse score when normalized!
balance_data: increases score on average
'''
X, y, X_no_labels = data.get(fill_na_strategy='mean', balance_data=True, verbose=True)

# splits randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

'''
Parameters:
 penalty (l1, l2), tol (tolerance), C, intercept_scaling, class_weight
 solver, max_iter, multi_class (ovr, multinomial), warm_start
 default is ok: dual, fit_intercept, random_state
 optional: verbose, n_jobs
'''
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)
print 'PE score: ', lr_classifier.score(X_test, y_test)

# Without PE
X, y, X_no_labels = data.get(without_PE=True, fill_na_strategy='mean', balance_data=True, verbose=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr_classifier.fit(X_train, y_train)
print 'No PE score: ', lr_classifier.score(X_test, y_test)