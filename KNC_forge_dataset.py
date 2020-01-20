import sys
# print("Python version:", sys.version)
import pandas as pd
# print("pandas version:", pd.__version__)
import matplotlib
import numpy as np
import scipy as sp
# import IPython
import sklearn
import mglearn


import matplotlib.pylab      as plt

from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split

X,y = mglearn.datasets.make_forge()

# Plot dataset
plt.figure(figsize=(10.,10.))
mglearn.discrete_scatter( X[:,0], X[:,1], y )
plt.legend(  ['class 0', 'class 1'], loc=4 )
plt.xlabel( '1st feature' )
plt.ylabel( '2nd feature' )
plt.show()

print( 'X shape', X.shape)
print( 'y shape', y.shape)

if(False):
	# n_neighbors=1
	mglearn.plots.plot_knn_classification(n_neighbors=1)
	plt.show()


if(False):
	# n_neighbors=3 
	mglearn.plots.plot_knn_classification(n_neighbors=3)
	plt.show()


if(False):
	# with train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	knc = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
	knc.fit(X_train, y_train)

	y_pred = knc.predict(X_test)
	print('Test set predictions: ', )
	print('Test ground truth: ', y_test)
	print('Test set accuracy: {:.2f}'.format( knc.score(X_test, y_test) ) )
	print( 'Accuracy of knc: {:.2f}'.format( 100.*sklearn.metrics.accuracy_score(y_test,y_pred) ) )


fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4) # fill colors
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()