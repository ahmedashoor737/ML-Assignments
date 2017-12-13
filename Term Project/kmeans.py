from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn import metrics
import data


#Functions Segment




#End of Functions Segment

X_train, Y_train, X_test = data.get(normalize_X=True, fill_na=True, verbose=True)

#Kmeans PARAMS
n_clusters = 9
init='k-means++'


km = KMeans(n_clusters=n_clusters, init=init)

km.fit(X_train)

clusters = defaultdict(list)
for i, label in enumerate(km.labels_):
    clusters[label].append((X_train[i],Y_train[i]))

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
        if Y_train[i] == 1:
            c1 += 1
        elif Y_train[i] == 2:
            c2 += 1
        elif Y_train[i] == 3:
            c3 += 1
        elif Y_train[i] == 4:
            c4 += 1
        elif Y_train[i] == 5:
            c5 += 1
        elif Y_train[i] == 6:
            c6 += 1
        elif Y_train[i] == 7:
            c7 += 1
        elif Y_train[i] == 8:
            c8 += 1
        elif Y_train[i] == 9:
            c9 += 9
    print "Class 1: {} - {}%\nClass 2: {} - {}%\nClass 3: {} - {}%\nClass 4: {} - {}%\nClass 5: {} - {}%\nClass 6: {} - {}%\nClass 7: {} - {}%\nClass 8: {} - {}%\nClass 9: {} - {}%\n\n\n\n".format(c1,round(c1*100/len(features),2),c2,round(c2*100/len(features),2),c3,round(c3*100/len(features),2),c4,round(c4*100/len(features),2),c5,round(c5*100/len(features),2),c6,round(c6*100/len(features),2),c7,round(c7*100/len(features),2),c8,round(c8*100/len(features),2),c9,round(c9*100/len(features),2))
print "Homogeneity: %0.3f" % metrics.homogeneity_score(Y_train, km.labels_)
print("Completeness: %0.3f" % metrics.completeness_score(Y_train, km.labels_))
print("V-measure: %0.3f\n\n\n" % metrics.v_measure_score(Y_train, km.labels_))
