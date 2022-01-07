import numpy as np

A = np.array([[1,2,-1],
              [-2,3,4],
              [3,1,2]])

b = np.array([[2],
              [-2],
              [3]])

def sigular_value_de(weightMatrix):
    # weightMatrix = self.weight_allLambda()
    U,S,VT = np.linalg.svd(weightMatrix,full_matrices=False)
    invert_W = np.dot(VT.T,np.dot(np.linalg.inv(np.diag(S)),U.T))
    return U,S,VT

x = np.matmul(np.linalg.inv(A),b)
print(x)

U,S,VT = sigular_value_de(A)
# print(np.matmul(U.T,b))
bprime = np.dot(np.dot(U.T,b),VT.T)