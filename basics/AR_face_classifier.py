#!python
#!/usr/bin/env python
from scipy.io import loadmat
x = loadmat('randomfaces/randomfaces4ar.mat')

# print(x['featureMat'])
print(x['featureMat'].shape)
print(x['filenameMat'].shape)
print(x['labelMat'].shape)
