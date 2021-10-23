import numpy as np
import sympy as sp
import tkinter
import matplotlib
import matplotlib.pyplot as plt
# DISPERSION

Layer_Data = np.array([[100,2],
                       [150,6],
                       [200,10],
                       [400,np.inf]])

# Define layer array (length N) and depth array (length N+1)
numLayers = len(Layer_Data[:,0])
Depth = np.append(0,Layer_Data[:,1])
# Shear wave velocity (Vs)
Vs = Layer_Data[:,0]
# Poisson'a ratio
PR = 0.3
# Disire wavelength
#Lambda = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])
Lambda = np.arange(2,44,3)
DC_points = len(Lambda)
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
	#Area = sp.integrate(PDF,(z,0,np.inf))
	A_term1 = (cv1/(cv3/WL))*(sp.exp(cv3/WL*np.inf) - sp.exp(cv3/WL*0))
	A_term2 = (cv2/(cv4/WL))*(sp.exp(cv4/WL*np.inf) - sp.exp(cv4/WL*0))
	Area = A_term1 + A_term2
	for j in range(num_layer):
		#Area_i[j] = sp.integrate(PDF,(z,limit_low[j],limit_up[j]))
		term1 = (cv1/(cv3/WL))*(sp.exp(cv3/WL*limit_up[j]) - sp.exp(cv3/WL*limit_low[j]))
		term2 = (cv2/(cv4/WL))*(sp.exp(cv4/WL*limit_up[j]) - sp.exp(cv4/WL*limit_low[j]))
		Area_i[j] = term1 + term2
		weights = Area_i / Area
	return weights
def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    inv_A_SVD = np.dot(V,np.dot(inv_s,UT))
    return U,s,VT,inv_A_SVD
# Compute weighting factor matrix
beta = (0.87+1.12*PR)/(1+PR)
Vph = np.zeros((len(Lambda)))
WEIGHTS = np.zeros((numLayers,len(Lambda)))
for i in range(len(Lambda)):
	weights = weighting(Depth,Lambda[i])
	# print(('({})-WEIGHTING = {}\nPHASE VELOCITY = {}').format(i+1,weights,Vph[i]))
	print('PASS LAMBDA [{}]'.format(i+1))
	WEIGHTS[:,i] = weights
	Vph[i] = np.dot(beta*Vs,weights.flatten())
print('weighting factor matrix\n',WEIGHTS.T)
print()
dispersiondata = np.zeros((len(Lambda),2))
dispersiondata[:,0],dispersiondata[:,1] = Lambda, Vph
print(dispersiondata)
dispersionDATA = np.savetxt('dispersiondata.txt',dispersiondata)
print(WEIGHTS.shape)
Vph = np.matmul(WEIGHTS.T,(0.93)*Vs)
U,s,VT,inv_A_SVD = SVD(WEIGHTS)
Vs = np.matmul(inv_A_SVD.T,(1/0.93)*Vph)
print(Vs)
