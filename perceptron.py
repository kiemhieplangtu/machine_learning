'''
https://machinelearningcoban.com/2017/01/21/perceptron/
PLA: Perceptron Learning Algorithm
1. Chọn ngẫu nhiên một vector hệ số w với các phần tử gần 0.
2. Duyệt ngẫu nhiên qua từng điểm dữ liệu xi:
   - Nếu xi được phân lớp đúng, tức sgn((w^T)xi)=yi, chúng ta không cần làm gì.
   - Nếu xi bị misclassifed, cập nhật w theo công thức:
           w=w+yixi
   - Kiểm tra xem có bao nhiêu điểm bị misclassifed. Nếu không còn điểm nào, dừng thuật toán. Nếu còn, quay lại bước 2.

'''

import numpy                as np
import matplotlib.pylab     as plt
import matplotlib.cm        as cm
from scipy.spatial.distance import cdist

def sgn(w, x):    
    return np.sign(np.dot(w.T, x))


def is_ok(x, w, y):
	return np.array_equal( sgn(w,x), y )



def draw_line(w):
    w0, w1, w2 = w[0], w[1], w[2]
    if w2 != 0.:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')
    else:
        x10 = -w0/w1
        return plt.plot([x10, x10], [-100, 100], 'k')


def plot(X, Nclass, label, colors, w=None):
	fig = plt.figure(figsize=(10,10) )

	for i,iy in enumerate([-1,1]):
		Xi = X[:, (label==iy)[0]]
		plt.plot(Xi[0,:], Xi[1,:], color=colors[i], marker='^', ls='None')

	# plt.plot(X[0,:], X[1,:], 'k.')
	if(w is not None):
		draw_line(w)

	plt.title('PLA: Perceptron Learning Algorithm, N_class = %i' %(Nclass) )
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.axis('equal')
	plt.xlim( np.min(X[0,:])-0.5, np.max(X[0,:])+0.5 )
	plt.ylim( np.min(X[1,:]), np.max(X[1,:]) )
	plt.show()

# np.random.seed(2)

means = [ [2., 2.], [4., 2.] ]
cov   = [ [0.3, 0.2], [0.2, 0.3] ]

N      = 10
Nclass = 2

colors = cm.rainbow( np.linspace(0., 1., Nclass) )

x0    = np.random.multivariate_normal( means[0], cov, N ).T
x1    = np.random.multivariate_normal( means[1], cov, N ).T

x     = np.concatenate( (x0, x1), axis=1 )
y     = np.concatenate( (np.ones((1,N)), -1*np.ones((1,N)) ), axis=1 )

# x_bar  là điểm dữ liệu mở rộng bằng cách thêm phần tử x0=1 lên trước vector x 
X     = np.concatenate( (np.ones( (1,2*N)), x), axis=0 )


# plot(x, Nclass, y, colors)


Npar    = X.shape[0]
Npoints = X.shape[1]
w       = np.random.rand(Npar,1)

print(w.shape)

while True:
	idx = np.random.permutation(Npoints)
	Xi  = X[:, idx]
	yi  = y[0, idx]
	for i in range(Npoints):
		xi = Xi[:,i].reshape(Npar,1)
		if( sgn(w,xi)[0] != yi[i] ):
			w = w + yi[i]*xi
	if( is_ok(X, w, y) ):
		break



plot(x, Nclass, y, colors, w)