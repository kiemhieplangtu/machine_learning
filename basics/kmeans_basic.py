from __future__ import print_function 
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(11)


def kmeans_display(X, label, centers):
	K = np.amax(label) + 1
	X0 = X[label == 0, :]
	X1 = X[label == 1, :]
	X2 = X[label == 2, :]

	plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
	plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
	plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

	for xci, yci  in centers:
		plt.plot(xci, yci, 'ks', markersize = 10, alpha = .8)

	plt.axis('equal')
	plt.plot()
	plt.show()



def kmeans_init_centers(X, Nclusters):
    # randomly pick Nclusters rows of X as initial centers
    return X[np.random.choice(X.shape[0], Nclusters, replace=False)]



def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)


def kmeans_update_centers(X, label, Nclusters):
	cen = np.zeros( (Nclusters, X.shape[1]) )
	for k in range(Nclusters):
		xk = X[ label==k, :]
		cen[k,:] = np.mean( xk, axis=0 )
	return cen



def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))



means = [ [2., 2.], [8., 3.], [3., 6.] ]
cov   = [ [1.5, 0.], [0., 1.5] ]
N     = 500
X0    = np.random.multivariate_normal(means[0], cov, N)
X1    = np.random.multivariate_normal(means[1], cov, N)
X2    = np.random.multivariate_normal(means[2], cov, N)

X     = np.concatenate((X0, X1, X2), axis = 0)
Nclusters  = 3

label = np.asarray([0]*N + [1]*N + [2]*N).T


centers     = kmeans_init_centers(X, Nclusters)
kmeans_display(X, label, centers)
# label       = original_label


i = 0
while True:
	print(i)
	
	label       = kmeans_assign_labels(X, centers)
	new_centers = kmeans_update_centers(X, label, Nclusters)	

	if( has_converged(centers, new_centers) ):
		print('OK')
		print(new_centers)
		print(label)
		break

	centers = new_centers
	i += 1


kmeans_display(X, label, new_centers)



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
kmeans_display(X, pred_label, kmeans.cluster_centers_)