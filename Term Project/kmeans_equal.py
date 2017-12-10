import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import normalize


#Functions Segment




#End of Functions Segment


#File names
fv = './facies_vectors.csv'
test_data = './test_data_nofacies.csv'

#Reading CSVs
fv_df = pd.read_csv(fv)
test_data_df = pd.read_csv(test_data)

#Dealing with NaNs
fv_df = fv_df.fillna(0)
test_data_df = test_data_df.fillna(0)

#Kmeans PARAMS
n_clusters = 9
init='k-means++'


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

fv_df = pd.concat(d)

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
#DataFrame Exploration
# print fv_df
# print 82 * '_'
# print test_data_df


X_train_df = fv_df[['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X_train = X_train_df.as_matrix()
X_train = normalize(X_train, axis=0)
Y_train_df = fv_df['Facies']
Y_train = Y_train_df.as_matrix()

X_test_df = test_data_df[['GR','ILD_log10','PE', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']]
X_test = X_test_df.as_matrix()
# Y_test_df = test_data_df['Facies']
# Y_test = Y_test_df.as_matrix()


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