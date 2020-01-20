import sys
# print("Python version:", sys.version)
import pandas as pd
# print("pandas version:", pd.__version__)

import numpy as np
import scipy as sp
import IPython
import sklearn


import matplotlib.pylab      as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
	                                stratify=cancer.target, random_state=66)

print(X_train.shape)
print(y_train.shape)
# print(cancer.feature_names)
print(cancer)


# data_pandas = pd.DataFrame(cancer)
# IPython.display(data_pandas)



if(False):
	test_acc  = []
	train_acc = []

	nrange = range(1,11)
	for n in nrange:
		knc = KNeighborsClassifier(n_neighbors=n)
		knc.fit(X_train, y_train)

		y_pred = knc.predict(X_test)
		print('Test set predictions: ', )
		print('Test ground truth: ', y_test)
		print('Test set accuracy: {:.2f}'.format( knc.score(X_test, y_test) ) )
		print( 'Accuracy of knc: {:.2f}'.format( 100.*sklearn.metrics.accuracy_score(y_test,y_pred) ) )

		test_acc.append( knc.score(X_test, y_test) )
		train_acc.append( knc.score(X_train, y_train) )


	plt.figure(figsize=(6., 6.) )
	plt.plot( nrange, test_acc, 'r-', label='test_acc')
	plt.plot( nrange, train_acc, 'k-', label='train_acc' )
	plt.xlabel( 'n_neighbors' )
	plt.ylabel( 'Accuracy' )
	plt.legend(loc=4)
	plt.show()