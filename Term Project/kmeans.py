from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import data

#Functions Segment
def kmeans_split(X,Y,percentage, km):
    km.fit(X,Y)

    clusters = defaultdict(list)
    for i, label in enumerate(km.labels_):
        clusters[label].append((X[i],Y[i]))

    for label, features in clusters.items():
        print "Cluster {}".format(label)
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        c4 = 0.0
        c5 = 0.0
        c6 = 0.0
        c7 = 0.0
        c8 = 0.0
        c9 = 0.0
        for i in range(0, len(features)):
            if Y[i] == 1:
                c1 += 1
            elif Y[i] == 2:
                c2 += 1
            elif Y[i] == 3:
                c3 += 1
            elif Y[i] == 4:
                c4 += 1
            elif Y[i] == 5:
                c5 += 1
            elif Y[i] == 6:
                c6 += 1
            elif Y[i] == 7:
                c7 += 1
            elif Y[i] == 8:
                c8 += 1
            elif Y[i] == 9:
                c9 += 9
        print "Class 1: {} - {}%\nClass 2: {} - {}%\nClass 3: {} - {}%\nClass 4: {} - {}%\nClass 5: {} - {}%\nClass 6: {} - {}%\nClass 7: {} - {}%\nClass 8: {} - {}%\nClass 9: {} - {}%\n\n\n\n".format(c1,round(c1*100/len(features),2),c2,round(c2*100/len(features),2),c3,round(c3*100/len(features),2),c4,round(c4*100/len(features),2),c5,round(c5*100/len(features),2),c6,round(c6*100/len(features),2),c7,round(c7*100/len(features),2),c8,round(c8*100/len(features),2),c9,round(c9*100/len(features),2))
    # print "Homogeneity: %0.3f" % metrics.homogeneity_score(Y, km.labels_)
    # print("Completeness: %0.3f" % metrics.completeness_score(Y, km.labels_))
    # print("V-measure: %0.3f\n\n\n" % metrics.v_measure_score(Y, km.labels_))
    return metrics.homogeneity_score(Y, km.labels_)


def km_use_kfolds(X,Y,n_splits,km):
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
    purity = []
    indices = []
    for train_in, test_in in kf.split(X):
        X = X[train_in]
        Y = Y[train_in]
        X_vald = X[test_in]
        Y_vald = Y[test_in]
        km.fit(X, Y)
        purity.append(metrics.homogeneity_score(Y_vald, km.labels_))
        indices.append([train_in[0],train_in[-1]])
    return purity,indices



#End of Functions Segment

X, Y, X_test = data.get(fill_na=True, show_class_distribution=True, verbose=True) #, normalize_X=True)

#Kmeans PARAMS
n_clusters = 9
init='k-means++'



print 'PE'
km = KMeans(n_clusters=n_clusters, init=init)

percentage = .9
reg_acc = kmeans_split(X,Y,percentage,km)
print 'Accuracy at {0:}% Seperation: {1:}\n'.format(percentage*100, reg_acc)
folds = 10
print 'Using {}-Folds'.format(folds)
kfold_purities, kfold_ind = km_use_kfolds(X,Y,folds,km)
print 'KFolds Purities: {0:},\nhighest: {1:.2f} - {2:}'.format(kfold_purities,kfold_purities[np.argmax(kfold_purities)], kfold_ind[np.argmax(kfold_purities)])
print 82 * '_'

print 'NO PE'

X, Y, X_test = data.get(without_PE=True, fill_na=True, verbose=True) #, normalize_X=True)


km = KMeans(n_clusters=n_clusters, init=init)

percentage = .9
reg_acc = kmeans_split(X,Y,percentage,km)
print 'Accuracy at {0:}% Seperation: {1:}\n'.format(percentage*100, reg_acc)
folds = 10
print 'Using {}-Folds'.format(folds)
kfold_purities, kfold_ind = km_use_kfolds(X,Y,folds,km)
print 'KFolds Purities: {0:},\nhighest: {1:.2f} - {2:}'.format(kfold_purities,kfold_purities[np.argmax(kfold_purities)], kfold_ind[np.argmax(kfold_purities)])
print 82 * '_'