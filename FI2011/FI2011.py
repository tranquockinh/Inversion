import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *
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
beta = (0.87+1.12*PR)/(1+PR)
# Disire wavelength
#Lambda = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])
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

def computeWeight(DC_points,numLayers,Depth):
    WEIGHTS = np.zeros((DC_points,numLayers))
    for i in range(DC_points):
        weights = weighting(Depth,Lambda[i])
        #Vph[i] = np.dot(Vs,weights.flatten()) * beta
        WEIGHTS[i,:] = weights
    return WEIGHTS
WEIGHTS = computeWeight(DC_points,numLayers,Depth)

def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    regetWeight = np.dot(U,np.dot(np.diag(s),VT))
    invertWeight = np.dot(V,np.dot(inv_s,UT))
    return regetWeight,invertWeight

def plotTable(weightMatrix,index = 'Lambda {}',columns = 'Layer {}'):
    lambdaLabels = []
    layerLabels = []
    for i in range(len(weightMatrix[:,0])):
        # lambdaLabels.append('Lambda {}'.format(i+1))
        lambdaLabels.append(index.format(i+1))
    for j in range(len(weightMatrix[0,:])):
        layerLabels.append(columns.format(j+1))
    print('\nForwad computation weight:\n', \
    DataFrame(weightMatrix, index = lambdaLabels, columns = layerLabels))

def newLayerArray(numLayerArray):
    newLayerSet = []
    #for i in range(len(Lambda) + 1):
    for i in range(numLayerArray+1):
        if i == 0:
            newLayerSet.append(i)
            newLayerSet.append(Lambda[i])
        elif i == (numLayerArray):
            newLayerSet.append(np.inf)
        else:
            newLayerSet.append(Lambda[i])
    return newLayerSet

def Forward(weightMatrix, Vs):
    Vph = []
    for i in range(len(weightMatrix[:,0])):
        Weightings = weightMatrix[i,:]
        Vph.append(np.dot(Weightings, Vs)*beta)
    fig,ax = plt.subplots(figsize=(5,6),dpi=100)
    ax.plot(Vph,Lambda,'-bo',markerfacecolor='None')
    ax.invert_yaxis()
    ax.set_xlabel('Phase velocity, Vph [m/s]')
    ax.set_ylabel('Wavelength, [m]')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # ax.set_xlim(0,Vs[-1])
    # ax.spines['bottom'].set_color('white')
    # ax.spines['right'].set_color('white')
    # plt.show()
    return Vph
Vph = Forward(WEIGHTS, Vs)

def initializing(Lambda):
    newWeightMatrix = np.zeros((len(Lambda),len(Lambda)))
    for i in range(len(Lambda)):
        if i == 0:
            newLayerSet = newLayerArray(i)
            newLayerSet[-1] = np.inf
        else:
            newLayerSet = newLayerArray(i)
        #for j in range(i+1):
        temp = weighting(newLayerSet,Lambda[i])
        newWeightMatrix[i,:i+1] = temp[0:]
        #newWeightMatrix[i][:i] = temp
    return newWeightMatrix

newWeightMatrix = initializing(Lambda)
plotTable(newWeightMatrix,index = 'Lambda {}',columns = 'Layer {}')

def Inversion(newWeightMatrix,Vph,Lambda):
    newLayerSet = newLayerArray(len(Lambda))
    regetWeight,invertWeight = SVD(newWeightMatrix)
    initialVs = (1/beta) * np.matmul(invertWeight,Vph)
    newProfile = np.zeros((len(initialVs),2))

    print('initializing stage, solve for initial Vs: \n', initialVs[:,np.newaxis][:])
    print((newLayerSet),len(initialVs),len(Lambda))
    return initialVs, newLayerSet
initialVs, newLayerSet = Inversion(newWeightMatrix,Vph,Lambda)
