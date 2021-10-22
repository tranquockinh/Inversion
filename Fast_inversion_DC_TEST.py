import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *
# DISPERSION
Layer_Data = np.array([[150,2],
                       [200,5],
                       [250,10],
                       [400,np.inf]])

# Arrange vectors of shear wave velocity and depth
numLayers = len(Layer_Data[:,0])
Depth = np.append(0,Layer_Data[:,1])
Vs = Layer_Data[:,0]
# Poisson's ratio
PR = 0.5
beta = (0.87+1.12*PR)/(1+PR)
#Lambda = np.array([2,5,10,100])
#Lambda = np.array([2,3,4,5,6,7,8,9,10,11,13,15,17,20,100])
Lambda = np.arange(2,44,3)
DC_points = len(Lambda)
# Function to compute weights
def weighting(D,WL):
    z = sp.symbols('z')
    material_coefficient = 1
    limit_low = D[:-1]
    limit_up = D[1:]
    cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi, -0.3933*2*np.pi])
    cv1 = cv[0]
    cv2 = cv[1]
    cv3 = cv[2]
    cv4 = cv[3]
    # Particle displacement function
    PDF = (cv1*sp.exp(cv3/WL*z) + cv2*sp.exp(cv4/WL*z))*\
    material_coefficient
    # Loop to compute wave a the wavelength given
    num_layer = len(D) - 1
    Area_i = np.zeros((num_layer),dtype='object')
    # Total area
    #Area = sp.integrate(PDF,(z,0,np.inf))
    A_term1 = (cv1/(cv3/WL))*(sp.exp(cv3/WL*np.inf) - sp.exp(cv3/WL*0))
    A_term2 = (cv2/(cv4/WL))*(sp.exp(cv4/WL*np.inf) - sp.exp(cv4/WL*0))
    Area = A_term1 + A_term2
    for j in range(num_layer):
        #Area_i[j] = sp.integrate(PDF,(z,limit_low[j],limit_up[j]))
        term1 = (cv1/(cv3/WL))*(sp.exp(cv3/WL*limit_up[j]) \
        - sp.exp(cv3/WL*limit_low[j]))
        term2 = (cv2/(cv4/WL))*(sp.exp(cv4/WL*limit_up[j]) \
        - sp.exp(cv4/WL*limit_low[j]))
        Area_i[j] = term1 + term2
    weights = Area_i / Area
    return weights
# Computing phase velocity by weighted average at each wavelength

def computeWeight(DC_points,numLayers,Depth,Vs):
    Vph = np.zeros((DC_points))
    WEIGHTS = np.zeros((DC_points,numLayers))
    for i in range(DC_points):
        weights = weighting(Depth,Lambda[i])
        #Vph[i] = np.dot(Vs,weights.flatten()) * beta
        WEIGHTS[i,:] = weights
    return WEIGHTS
WEIGHTS = computeWeight(DC_points,numLayers,Depth,Vs)

def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    regetWeight = np.dot(U,np.dot(np.diag(s),VT))
    invertWeight = np.dot(V,np.dot(inv_s,UT))
    return regetWeight,invertWeight

def plotTable(weightMatrix):
    lambdaLabels = []
    layerLabels = []
    for i in range(len(weightMatrix[:,0])):
        lambdaLabels.append('Lambda {}'.format(i+1))
    for j in range(len(weightMatrix[0,:])):
        layerLabels.append('Layer {}'.format(j+1))
    print('\nForwad computation weight:\n', \
    DataFrame(weightMatrix, index = lambdaLabels, columns = layerLabels))

def newLayerArray(numLayerArray):
    newLayerSet = []
    #for i in range(len(Lambda) + 1):
    for i in range(numLayerArray+1):
        if i == 0:
            newLayerSet.append(i)
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
    fig,ax = plt.subplots(figsize=(4,5),dpi=100)
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

Vph = Forward(WEIGHTS,Vs)
def CheckPoint(numLayerRefine, Vph):
    numWavelength = numLayerRefine
    layerRefineSet = newLayerArray(numLayerRefine)
    useVph = []
    convertVs = []
    for i in range(numLayerRefine):
        useVph.append(Vph[i])
        convertVs.append(Vph[i] * (1/beta))
    newWeightMatrix = computeWeight(numLayerRefine,numWavelength,\
    layerRefineSet,convertVs)
    plotTable(newWeightMatrix)
    regetWeight,invertWeight = SVD(newWeightMatrix)
    newVs = np.matmul(invertWeight, convertVs)
    return newWeightMatrix,newVs,useVph
newWeightMatrix,newVs,useVph = CheckPoint(2, Vph)
print(newWeightMatrix)


# numLayers_refine = 6
# Bottoms = []
# for i in range(numLayers_refine):
#     Coeff = 1 # NEARLY 1/3 OF WAVELENGTH
#     Current_bottom = Coeff*Lambda[i]
#     Bottoms = np.append(Bottoms,Current_bottom)
# Bottoms[-1] = np.inf
# Depths = np.append(0,Bottoms)
#
# # DO NOT CHANGE # LAYERS AND THICKNESSES
# # COMPUTE WEIGHTS AT EACH WAVELENGTH
# WEIGHT_MATRIX = np.zeros((numLayers_refine,numLayers_refine))
# for i in range(numLayers_refine):
#     weights = weighting(Depths,Lambda[i])
#     WEIGHT_MATRIX[:,i] = weights
#     print(('WAVELENGTH {} PASSED').format(i+1))
# print(WEIGHT_MATRIX)


# # Show weight matrix
# fig,ax = plt.subplots(figsize=(2,1))
# ax.matshow(WEIGHT_MATRIX, cmap=plt.cm.Blues)
# for i in range(numLayers_refine):
#     for j in range(numLayers_refine):
#         text = ('{0:.3f}').format(WEIGHT_MATRIX[i,j])
#         ax.text(j, i, str(text), va='center', ha='center')
# plt.show()

#
# U,s,VT,inv_A_SVD = SVD(WEIGHT_MATRIX)
# RE_CONSTRUCTION = np.matmul(inv_A_SVD.T,Vph[:numLayers_refine]*(1/beta))
# print()
# print(RE_CONSTRUCTION)
#
# figure, ax = plt.subplots(figsize=(3,5))
# plotx = np.append(RE_CONSTRUCTION,RE_CONSTRUCTION[-1])
# ploty = np.append(Depths[:numLayers_refine],Depths[:numLayers_refine][-1]+5)
# ax.step(plotx,ploty)
# ax.invert_yaxis()
# ax.set_xlabel('Shear wave velocity, Vs [m/s]')
# ax.set_ylabel('Depth, z [m]')
# plt.show()
