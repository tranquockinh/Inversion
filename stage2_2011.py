import sympy as sp
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from weighting_func import weighting

dispersionDATA = np.loadtxt('dispersiondata.txt', dtype='float', delimiter=None)
Lambda,Vph = dispersionDATA[:,0],dispersionDATA[:,1]
weights = np.loadtxt('Weight_matrix.txt', dtype='float', delimiter=None)
invW = np.linalg.inv(weights)

Depth_convert = np.append(0,0.3*Lambda)
Depth_convert[-1] = np.inf

# Solve for initial profile
Vs_init = np.matmul(invW,(1/0.93)*Vph)


def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    inv_A_SVD = np.dot(V,np.dot(inv_s,UT))
    return U,s,VT,inv_A_SVD

# Now! time to compute the dispersion curve with new shear
# wave velocity profile
W_updated = np.zeros((len(Lambda),len(Lambda)))
for i in range(len(Lambda)):
    weights_updated = weighting(Depth_convert,Lambda[i])
    W_updated[i,:] = weights_updated

# Here we have a initial model based on weighted.
# We may want to see how the dispersion curve look like.

Vph_ADJUSTED = 0.8*Vph+0.2*0.93*Vs_init
Vs_updated = np.matmul(invW,(1/0.93)*Vph_ADJUSTED)
# check if new shear wave based on new Vs move closer to the true DC curve
Vph_updated = np.matmul(weights,Vph_ADJUSTED)
R = Vph_updated - Vph
print(R)

fig,ax = plt.subplots(figsize=(4,5),dpi=100)
ax.step(np.append(Vs_init,Vs_init[-1]),0.3*np.append(Lambda,(Lambda[-1]+3)))
ax.step(np.append(Vs_updated,Vs_updated[-1]),0.3*np.append(Lambda,(Lambda[-1]+3)))

ax.invert_yaxis()
ax.set_xlabel('Shear wave velocity, Vs [m/s]')
ax.set_ylabel('Depth, z [m]')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

fig,ax = plt.subplots(figsize=(4,5),dpi=100)
ax.plot(Vph,Lambda,'-bo',markerfacecolor='None')
ax.plot(Vph_updated,Lambda,'-ro',markerfacecolor='None')
ax.invert_yaxis()
ax.set_xlabel('Phase velocity, Vph [m/s]')
ax.set_ylabel('Wavelength, [m]')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
plt.show()


'''
num_iter = 5
def adjust(Vs_old,Vs_new):
	Vs_adjust = 0.8*Vs_old + 0.2*Vs_new
	return Vs_adjust

R = np.zeros((num_iter),dtype = object)
Vs_old = (1/0.93) * Vph
for i in range(num_iter):
	if i != 0:
		R[i] = np.zeros((len(Vph[1:])))
		# Compute new array of shear wave velocity
		Vs_new = np.matmul(invW[1:,1:],Vs_old[1:])
		# Compute difference of new and old shear wave velocity
		R[i] = Vs_new - Vs_old[1:]
		# Adjust new shear wave velocity
		Vs_adjust = adjust(Vs_old[1:],Vs_new)
		Vs_old[1:] = Vs_new
		print(Vs_adjust)
'''
