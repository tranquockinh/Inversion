#Tocken: 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
import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *

PR = 0.3 # Poisson'a ratio
beta = (0.87+1.12*PR)/(1+PR)

# Layer_Data = np.array([[150,5],
#                        [400,np.inf]])
# Layer_Data = np.array([[150,2],
                       # [200,5],
                       # [250,10],
                       # [400,np.inf]])
                       
Layer_Data = np.array([[150,5],
                       [400,np.inf]])
waveLength0 = 6
deltaWaveLength = 6
Lambda = np.arange(waveLength0,45,deltaWaveLength)

def dataInput(soilProfile, desiredWavelength):
    numLayers = len(soilProfile[:,0])
    DC_points = len(desiredWavelength)
    Depth =np.append(0, soilProfile[:,1])
    Vs = soilProfile[:,0]
    return numLayers,Depth,Vs,DC_points
numLayers,Depth,Vs,DC_points = dataInput(Layer_Data,Lambda)

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
	return Area_i/Area_i[0],weights

Area_i,weights = weighting(Depth,5)
print(Area_i)