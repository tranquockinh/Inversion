import numpy as np
import sympy as sp
import tkinter
import matplotlib
import matplotlib.pyplot as plt
# DISPERSION

Layer_Data = np.array([[200,5],
                       [300,10],
                       [400,np.inf]])

# Model parameters
nLayers = len(Layer_Data[:,0])
Depth = np.append(0,Layer_Data[:,1])
Vs = Layer_Data[:,0]
# Convert shear wave velocity to phase velocity
nuy = 0.3
fv = (0.87 + 1.12*nuy) / (1 + nuy)
Vph_converted = Layer_Data[:,0] * (fv)
# Assume a range of wavelength Lambda
Lambda = sp.Matrix([10,11,12,13,14,15])

# Compute weighting factor using weighting function
def weighting(D,Lambdai):
    D[-1] = Lambda[-1]
    nLayers = len(D) - 1
    limit_low = D[:-1]
    limit_up = D[1:]

    z = sp.symbols('z')
    PDF = 1 - (z/Lambdai) ** (3/2)
    int_PDF = z - (2/5) * ((z/Lambdai)**(5/2)) * Lambdai
    Area_i = np.zeros((nLayers),dtype='object')
    # Total area
    Area = int_PDF.subs({z:Lambdai}) - int_PDF.subs({z:0})
    for j in range(nLayers):
        Area_i[j] = int_PDF.subs({z:limit_up[j]}) - int_PDF.subs({z:limit_low[j]})
        weights = Area_i / Area
    return weights
# Compute weighting factor matrix
Vph = np.zeros((len(Lambda)))
WEIGHTS = np.zeros((nLayers,len(Lambda)))
for i in range(len(Lambda)):
	weights = weighting(Depth,Lambda[i])
	print('PASS LAMBDA [{}]'.format(i+1))
	WEIGHTS[:,i] = weights
	Vph[i] = np.dot(fv*Vs,weights.flatten())
print()
print('weighting factor matrix\n',WEIGHTS.T)

