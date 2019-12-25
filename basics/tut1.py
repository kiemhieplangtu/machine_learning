import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('advertising.csv')

# print df

x = df.values[:,2]
y = df.values[:,4]

print type(x)
print x.shape

if(False):
	plt.scatter(x,y, marker='o', c='k')
	plt.plot(x, x*0.48 + 0.0014, 'r-')
	plt.plot(x, x*0.4820883 + 0.17474956, 'b-')
	plt.show()

	sys.exit()

def predict(x, a, b):
	return a*x + b

def cost_fcn(x, y, a, b):
	n = len(x)
	s = (y - (a*x + b))**2
	s = s.sum()
	return s/n


def update(x, y, a, b, rate):
	n = len(x)

	t1 = -2.*x*(y-(a*x+b))
	t1 = t1.sum()
	t1 = t1/n

	t2 = -2.*(y-(a*x+b))
	t2 = t2.sum()
	t2 = t2/n

	a -= t1*rate
	b -= t2*rate

	return a,b

def train(x, y, a, b, rate, loop):
	cos_his = []
	for i in range(loop):
		a, b = update(x, y, a, b, rate)
		cost = cost_fcn(x, y, a, b)
		cos_his.append(cost)

	return a, b, cos_his

a, b, cos_arr = train(x, y, 0.03, 0.0014, 0.001, 60)

print a
print b

print ''
print cos_arr

print ''
print predict(19., a, b)

nloops = range(60)
plt.plot(nloops, cos_arr)
plt.show()