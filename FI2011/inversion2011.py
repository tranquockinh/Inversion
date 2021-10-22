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
	
W = np.zeros((len(Lambda),len(Lambda)))
for i in range(len(Lambda)):
	Depth_array = 0.3*Lambda[:i+1]
	Depth = np.append(0,Depth_array)
	Depth[-1] = np.inf
	weights = weighting(Depth,Lambda[i])
	W[i,:i+1] = weights
	if i != (len(Lambda)-1):	
		print('------------------------> ok')
	else:
		print(('------------------------> finished'))

np.savetxt('Weight_matrix.txt',W)
print('weighting factor matrix saved with the name: "Weight_matrix"')

