#Tocken: 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

import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *

PR = 0.3 # Poisson'a ratio
beta = (0.87+1.12*PR)/(1+PR)
Layer_Data = np.array([[150,2],
                       [200,5],
                       [250,10],
                       [400,np.inf]])
Lambda = np.arange(2,54,3)

def dataInput(soilProfile, desiredWavelength):
    numLayers = len(soilProfile[:,0])
    DC_points = len(desiredWavelength)
    Depth =np.append(0, soilProfile[:,1])
    Vs = soilProfile[:,0]
    return numLayers,Depth,Vs,DC_points
numLayers,Depth,Vs,DC_points = dataInput(Layer_Data,Lambda)

def weighting(D,WL):
	z = sp.symbols('z')
	limit_low = D[:-1]
	limit_up = D[1:]
	material_coefficient = 1
	cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi,-0.3933*2*np.pi])
	cv1,cv2,cv3,cv4 = cv[0],cv[1],cv[2],cv[3]
	PDF = (cv1*sp.exp(cv3/WL*z) + cv2*sp.exp(cv4/WL*z))*material_coefficient
	num_layer = len(D) - 1
	Area_i = np.zeros((num_layer),dtype='object')
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
    print('\nForwad computation:\n', \
    DataFrame(weightMatrix, index = lambdaLabels, columns = layerLabels))

def Forward(weightMatrix, Vs):
    Vph = []
    for i in range(len(weightMatrix[:,0])):
        Weightings = weightMatrix[i,:]
        Vph.append(np.dot(Weightings, Vs)*beta)
    return Vph
Vph = Forward(WEIGHTS, Vs)

def newLayerArray(numLayerArray):
    Alpha = 0.3
    convertDepth = Alpha * Lambda
    newLayerSet = []
    for i in range(numLayerArray):
        if i == 0:
            newLayerSet.append(i)
            newLayerSet.append(convertDepth[i])
        elif i == (numLayerArray - 1):
            newLayerSet.append(np.inf)
        else:
            newLayerSet.append(convertDepth[i])
    return newLayerSet
newLayerSet = newLayerArray(len(Lambda))

def initializing(Lambda):
    newWeightMatrix = np.zeros((len(Lambda),len(Lambda)))
    for i in range(len(Lambda)):
        if i == 0:
            newLayerSet = newLayerArray(i+1)
        else:
            newLayerSet = newLayerArray(i+1)
        newLayerSet[-1] = np.inf
        temp = weighting(newLayerSet,Lambda[i])
        newWeightMatrix[i,:i+1] = temp[0:]
    return newWeightMatrix
newWeightMatrix = initializing(Lambda)
plotTable(newWeightMatrix,index = 'Lambda {}',columns = 'Layer {}')

def InversionInit(newWeightMatrix,Vph,Lambda):
    newLayerSet = newLayerArray(len(Lambda))
    regetWeight,invertWeight = SVD(newWeightMatrix)
    initialVs = (1/beta) * np.matmul(invertWeight,Vph)
    Data = np.zeros((len(initialVs),4))
    convertVs,initDataVs,Difference,initDataDepth = [],[],[],[]
    for i in range(len(initialVs)):
        convertVs.append(Vph[i] * (1/beta))
        initDataVs.append(initialVs[i])
        Difference.append(initDataVs[i] - convertVs[i])
        initDataDepth.append(newLayerSet[1:][i])
    Data[:,0] = initDataVs
    Data[:,1] = initDataDepth
    Data[:,2] = convertVs
    Data[:,3] = Difference
    initialDataProfile = Data[:,:2]
    plotTable(Data,index = 'Layer depth {}',columns = 'Data {}')
    return initialDataProfile
initialDataProfile = InversionInit(newWeightMatrix,Vph,Lambda)

def TestDispersion(Vph,initialDataProfile):
    numLayers,Depth,Vs,DC_points = dataInput(initialDataProfile,Lambda)
    WeightingMatrix = computeWeight(DC_points,numLayers,Depth)
    regetWeight,invertWeight = SVD(WeightingMatrix)
    reducedVs = []
    for i in range(len(Vs)):
        reducedVs.append(0.2 * Vs[i] + 0.8 * (1/beta) * Vph[i])
    ShearWaveVeloc = Forward(WeightingMatrix, reducedVs)
    print(ShearWaveVeloc)
    # plotTable(ShearWaveVeloc,index = 'Lambda {}',columns = 'Layer {}')
    return ShearWaveVeloc,initialDataProfile
ShearWaveVeloc,initialDataProfile = TestDispersion(Vph,initialDataProfile)


def plot():
    fig,ax = plt.subplots(figsize=(5,6),dpi=100)
    numLayers,Depth,Vs,DC_points = dataInput(Layer_Data,Lambda)
    VsPlot = np.append(Vs,Vs[-1])
    Depth[-1] = Depth[-2] + 5
    ax.step(VsPlot,Depth)
    numLayers,Depth,Vs,DC_points = dataInput(initialDataProfile,Lambda)
    VsPlot = np.append(Vs,Vs[-1])
    Depth[-1] = Depth[-2] + 1
    ax.step(VsPlot,Depth)
    
    '''
    ax.plot(Vph,newLayerSet[1:],'-bo',markerfacecolor='None')
    ax.plot(ShearWaveVeloc,newLayerSet[1:],'-ro',markerfacecolor='None')
    '''
    ax.invert_yaxis()
    ax.set_xlabel('Phase velocity, Vph [m/s]')
    ax.set_ylabel('Wavelength, [m]')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # ax.set_xlim(0,Vs[-1])
    # ax.spines['bottom'].set_color('white')
    # ax.spines['right'].set_color('white')
    plt.show()
plot()
