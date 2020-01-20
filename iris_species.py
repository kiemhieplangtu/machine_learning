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


def myweight(distances):
    sigma2 = .5          # we can change this number
    return np.exp(-distances**2/sigma2)




dat = load_iris()

print('Keys of iris datasets:\n', dat.keys())
# print(dat['DESCR'][:593] + '\n...')
# print(dat['data'] )

X_train, X_test, y_train, y_test = train_test_split( dat['data'], dat['target'], random_state=0, test_size=50 ) #test_size=50

# print('X_test', X_test.shape)
# print('y_test', y_test.shape)

if(False):
	# create dataframe from data in X_train
	# label the columns using the strings in iris_dataset.feature_names
	datframe = pd.DataFrame( X_train, columns=dat.feature_names )

	# create a scatter matrix from the dataframe, color by y_train
	pd.plotting.scatter_matrix( datframe, c=y_train, figsize=(10,10),
	                            marker='o', hist_kwds={'bins':20}, s=60, alpha=0.8, cmap=mglearn.cm3 )
	plt.show()


# with n_neighbors = 1
if(False):
	knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, p=2) # p=2 for Euclidean distance
	knn.fit( X_train, y_train )

	y_pred = knn.predict(X_test)

	print( 'Test set prediction: \n', y_pred )
	print( 'Test true labels:\n', y_test)

	print( 'Test score: {:.2f}'.format(np.mean(y_pred==y_test)) )
	print( 'Test score: {:.2f}'.format(knn.score(X_test, y_test))  )
	print( 'Accuracy of k1nn: {:.2f}'.format( 100.*sklearn.metrics.accuracy_score(y_test,y_pred) ) )

# With n_neighbors = 10
if(False):
	knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, p=2) # p=2 for Euclidean distance
	knn.fit( X_train, y_train )

	y_pred = knn.predict(X_test)

	print( 'Test set prediction: \n', y_pred )
	print( 'Test true labels:\n', y_test)

	print( 'Test score: {:.2f}'.format(np.mean(y_pred==y_test)) )
	print( 'Test score: {:.2f}'.format(knn.score(X_test, y_test))  )
	print( 'Accuracy of k1nn: {:.2f}'.format( 100.*sklearn.metrics.accuracy_score(y_test,y_pred) ) )



# With n_neighbors = 10 and  weights = 'distance'
if(False):
	print('')
	print( 'n_neighbors = 10 and  weights = distance' )
	knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights = 'distance') # p=2 for Euclidean distance
	knn.fit( X_train, y_train )

	y_pred = knn.predict(X_test)

	print( 'Test set prediction: \n', y_pred )
	print( 'Test true labels:\n', y_test)

	print( 'Test score: {:.2f}'.format(np.mean(y_pred==y_test)) )
	print( 'Test score: {:.2f}'.format(knn.score(X_test, y_test))  )
	print( 'Accuracy of k1nn: {:.2f}'.format( 100.*sklearn.metrics.accuracy_score(y_test,y_pred) ) )




# With n_neighbors = 10 and  weights = 'myweights'
if(True):
	print('')
	print( 'n_neighbors = 10 and  weights = myweight' )
	knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights = myweight) # p=2 for Euclidean distance
	knn.fit( X_train, y_train )

	y_pred = knn.predict(X_test)

	print( 'Test set prediction: \n', y_pred )
	print( 'Test true labels:\n', y_test)

	print( 'Test score: {:.2f}'.format(np.mean(y_pred==y_test)) )
	print( 'Test score: {:.2f}'.format(knn.score(X_test, y_test))  )
	print( 'Accuracy of k1nn: {:.2f}'.format( 100.*sklearn.metrics.accuracy_score(y_test,y_pred) ) )