import numpy as np

a=8
b=7
matrx = np.zeros((a**b,a+1))

seq = [1,2,3,4,5,6,7,8]
for i in range(matrx.shape[1]-1):
    matrx[:,i] = np.tile(np.repeat(seq, matrx.shape[0]/(len(seq)**(i+1))),(i+1))