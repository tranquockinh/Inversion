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
W = np.zeros((len(Lambda),len(Lambda)))
for i in range(len(Lambda)):
	Depth_array = Lambda[:i+1]
	Depth = np.append(0,Depth_array)
	Depth[-1] = np.inf
	weights = weighting(Depth,Lambda[i])
	W[i,:i+1] = weights

def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    inv_SVD = np.dot(V,np.dot(inv_s,UT))
    return inv_SVD

inv_SVD = SVD(W)
print((1/beta)*np.matmul(inv_SVD,Vph))
#
# W_updated = np.zeros((len(Lambda),len(Lambda)))
# for i in range(len(Lambda)):
#     weights_updated = weighting(Depth_convert,Lambda[i])
#     #W_updated[i,:] = weights_updated
'''
	#print(Depth)
	if i != (len(Lambda)-1):
		print('------------------------> ok')
	else:
		print(('------------------------> finished'))
np.savetxt('Weight_matrix.txt',W)
'''
'''
fig,ax = plt.subplots(figsize=(4,5),dpi=100)
ax.plot(Vph,Lambda,'-bo',markerfacecolor='None')
ax.invert_yaxis()
ax.set_xlabel('Phase velocity, Vph [m/s]')
ax.set_ylabel('Wavelength, [m]')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
plt.show()
'''
