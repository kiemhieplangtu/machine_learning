import sys
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.cm     as cm

from scipy.spatial.distance import cdist

def init_centers(Ncen, Nclus, X, Npoints):
	# Choose centroids randomly (Ncen may be diferent from Nclus)
	sel_clus = np.random.choice(Nclus, Ncen, replace=False)

	# Choose a center point in each cluster
	cen = np.zeros( (Ncen, 2) )
	for i,iclus in enumerate(sel_clus):
		ip       = np.random.choice(Npoints, 1, replace=False)[0]
		cen[i,:] = X[ip,:,iclus]

	return cen


def update_centers(X, Ncen, label):
	c = np.zeros( (Ncen, 2) )
	for i in range(Ncen):
		c[i,:] = np.mean( X[label==i,:], axis=0)

	return c


def set_labels(X, centers):
	return cdist(centers, X).argmin(axis=0)


def has_converged(cen, new_cen):
	return set( tuple(a) for a in cen ) == set( tuple(a) for a in new_cen )


def plot(X, Nclus, Ncen, centers, label, colors):
	fig = plt.figure(figsize=(10,10) )

	# for i in range(Nclus):
	# 	plt.plot(XX[:,0,i], XX[:,1,i], 'k.')

	for i in range(Ncen):
		Xi = X[label==i,:]
		plt.plot(Xi[:,0], Xi[:,1], color=colors[i], marker='.', ls='None')

	# plt.plot(X[:,0], X[:,1], 'k.')

	for i,(xc,yc) in enumerate(centers):
		plt.plot(xc, yc, '^', color=colors[i], markersize=12, markeredgecolor='k', markeredgewidth=1.5)

	plt.title('K-means Clustering, N_clusters = %i, N_centroids = %i' %(Nclus, Ncen) )
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.axis('equal')
	plt.show()


## Generate clusters and data points
# np.random.seed(1)
Ncen    = 4
Nclus   = 5
Npoints = 500
colors  = cm.rainbow( np.linspace(0., 1., Ncen) )

# xlabl = []
# for i in range(Ncen):
# 	xlabl += [i]*Npoints

# xlabl = np.array( xlabl )

x_      = np.random.randint(low=1, high=80, size=Nclus)
y_      = np.random.randint(low=1, high=80, size=Nclus)

# aa = 100.*np.random.rand(5) 
# print("1D Array filled with random values : \n", aa)

mean = []
for (xm, ym) in zip(x_, y_):
	mean.append( [xm, ym] )

cov = [ [5., 0.], [0., 5.]]  # diagonal covariance

XX   = np.zeros( (Npoints, 2, Nclus) )
for i,xi in enumerate(mean):
	XX[:,:, i] = np.random.multivariate_normal(xi, cov, Npoints)

X = XX[:,:, 0]
for i in range(1, Nclus):
	X = np.concatenate( (X, XX[:,:,i]), axis=0 )



cen   = init_centers(Ncen, Nclus, XX, Npoints)

while True:
	xlabl   = set_labels(X, cen)

	new_cen = update_centers(X, Ncen, xlabl)
	xlabl   = set_labels(X, new_cen)

	if( has_converged(cen, new_cen) ):
		print('OK')
		print(new_cen)
		print(xlabl)
		plot(X, Nclus, Ncen, new_cen, xlabl, colors)
		break

	cen = new_cen