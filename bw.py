import sympy as sp
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

dispersionDATA = np.loadtxt('dispersiondata.txt', dtype='float', delimiter=None)
Lambda,Vph = dispersionDATA[:,0],dispersionDATA[:,1]


# Coefficients
PR = 0.3
beta = (0.87+1.12*PR)/(1+PR)

# Compute weight factors
def weighting(D,WL):
	# Integral variable and limits
	z = sp.symbols('z')
	limit_low = D[:-1]
	limit_up = D[1:]
	material_coefficient = 1
	# Coefficients of vertical particle displacement function
	cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi,-0.3933*2*np.pi])
	cv1,cv2,cv3,cv4 = cv[0],cv[1],cv[2],cv[3]
	# Particle displacement function
	PDF = (cv1*sp.exp(cv3/WL*z) + cv2*sp.exp(cv4/WL*z))*material_coefficient
	# Loop to compute weight at the wavelength given
	num_layer = len(D) - 1
	Area_i = np.zeros((num_layer),dtype='object')
	# Total area
	Area = sp.integrate(PDF,(z,0,np.inf))
	for j in range(num_layer):
		Area_i[j] = sp.integrate(PDF,(z,limit_low[j],limit_up[j]))
		weights = Area_i / Area
	return weights
	
	
# Singular value decomposition
def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    inv_A_SVD = np.dot(V,np.dot(inv_s,UT))
    return U,s,VT,inv_A_SVD
    
    
# #-Layer model
numLayers_refine = len(Lambda)
Bottoms = []
for i in range(numLayers_refine):
    Coeff = 1 
    Current_bottom = Coeff*Lambda[i]
    Bottoms = np.append(Bottoms,Current_bottom)
Bottoms[-1] = np.inf # assume half-space
Depths = np.append(0,Bottoms) # surface 

WEIGHT_MATRIX = np.zeros((numLayers_refine,len(Lambda)))
for i in range(len(Lambda)):
    weights = weighting(Depths,Lambda[i])
    WEIGHT_MATRIX[:,i] = weights
    print(('WAVELENGTH {} PASSED').format(i+1))
print(WEIGHT_MATRIX.T)

# Reconstruction 
U,s,VT,inv_A_SVD = SVD(WEIGHT_MATRIX)
Vs = np.matmul(inv_A_SVD.T,Vph*(1/beta))
print()
print(Vs)

