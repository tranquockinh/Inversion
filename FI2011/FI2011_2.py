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

def plotTable(weightMatrix,index ='Lambda {}',columns='Layer {}',Name = 'Table name'):
    lambdaLabels = []
    layerLabels = []
    for i in range(len(weightMatrix[:,0])):
        lambdaLabels.append(index.format(i+1))
    for j in range(len(weightMatrix[0,:])):
        layerLabels.append(columns.format(j+1))
    df = DataFrame(weightMatrix,index=Index(lambdaLabels, name=Name),columns=layerLabels)
    df.style
    print(df)

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
        elif i == (numLayerArray - 1): newLayerSet.append(np.inf)
        else: newLayerSet.append(convertDepth[i])
    return newLayerSet
newLayerSet = newLayerArray(len(Lambda))

def initializing(Lambda):
    newWeightMatrix = np.zeros((len(Lambda),len(Lambda)))
    for i in range(len(Lambda)):
        if i == 0:
            newLayerSet = newLayerArray(i+1)
        else: newLayerSet = newLayerArray(i+1)
        newLayerSet[-1] = np.inf
        temp = weighting(newLayerSet,Lambda[i])
        newWeightMatrix[i,:i+1] = temp[0:]
    return newWeightMatrix
newWeightMatrix = initializing(Lambda)


# Print out the weight matrix of forward computation and the weight matrix in the first stage of inversion.
print('\n\nForward computation -- Start from known Vs-profile -- Weight Matrix:')
plotTable(WEIGHTS,index = 'Lambda {}',columns = 'Layer {}',Name='Weight Matrix')
print('\n\nStage-1: Solve for Vs from top to bottom -- Weight Matrix:')
plotTable(newWeightMatrix,index = 'Lambda {}',columns = 'Layer {}',Name='Weight Matrix')

def InversionInit(newWeightMatrix,Vph,Lambda):
    newLayerSet = newLayerArray(len(Lambda))
    regetWeight,invertWeight = SVD(newWeightMatrix)
    initialVs = (1/beta) * np.matmul(invertWeight,Vph)
    initialDataProfile = np.zeros((len(initialVs),2))
    initDataVs = []
    for i in range(len(initialVs)):
        initDataVs.append(initialVs[i])
    initialDataProfile[:,0] = initDataVs
    initialDataProfile[:,1] = newLayerSet[1:]
    # plotTable(Data,index ='Layer depth {}',columns='Data {}',Name='Data for adjusting')
    return invertWeight,initialDataProfile
invertWeight,initialDataProfile = InversionInit(newWeightMatrix,Vph,Lambda)

def cutOffData(newWeightMatrix,Vph,Lambda):
    invertWeight,initialDataProfile = InversionInit(newWeightMatrix,Vph,Lambda)
    numLayers,Depth,Vs,DC_points = dataInput(initialDataProfile,Lambda)
    firstLayerVs = initialDataProfile[0,0]
    willAdjustVs = initialDataProfile[:,0]
    invertWeightCutOff = invertWeight[1:,:]
    # plotTable(invertWeightCutOff,index = 'Layer depth {}',columns = 'Data {}',Name='Invert weight cutoff')
    return firstLayerVs,willAdjustVs,invertWeightCutOff

def TestDispersion(newWeightMatrix,Vph,Lambda):
    firstLayerVs,willAdjustVs,invertWeightCutOff = cutOffData(newWeightMatrix,Vph,Lambda)
    Depth = newLayerSet[1:]
    ShearWaveVeloc = []
    dataSecondStage = np.zeros((len(Lambda),2))
    fromSecondVs = np.matmul(invertWeightCutOff,willAdjustVs)
    for i in range(len(Lambda)):
        if i == 0: ShearWaveVeloc.append(firstLayerVs)
        else: ShearWaveVeloc.append(fromSecondVs[i-1])
        dataSecondStage[i,0] = ShearWaveVeloc[i]
        dataSecondStage[i,1] = Depth[i]
    print('\n\nFirst iteration in the second stage: Adjusting the shear wave velocity to predict dispersion curve')
    plotTable(dataSecondStage,index = 'Layer depth {}',columns = 'data {}',Name='Shear wave velo.')
    return dataSecondStage
dataSecondStage = TestDispersion(newWeightMatrix,Vph,Lambda)
# print(dataSecondStage[:,0] - initialDataProfile[:,0])

def plot():
    fig,ax = plt.subplots(1,2,figsize=(8,6),dpi=100)
    numLayers,Depth,Vs,DC_points = dataInput(Layer_Data,Lambda)
    VsPlot = np.append(Vs,Vs[-1])
    Depth[-1] = Depth[-2] + 5
    ax[0].step(VsPlot,Depth)
    numLayers,Depth,Vs,DC_points = dataInput(initialDataProfile,Lambda)
    VsPlot = np.append(Vs,Vs[-1])
    Depth[-1] = Depth[-2] + 1
    # ax[0].step(VsPlot,Depth)
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Phase velocity, Vph [m/s]')
    ax[0].set_ylabel('Depth, [m]')
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()
    ax[0].spines['bottom'].set_color('white')
    ax[0].spines['right'].set_color('white')
    ax[1].plot(Vph,Lambda,'-bo',markerfacecolor='None',label='DC')
    numLayers,Depth,Vs,DC_points = dataInput(initialDataProfile,Lambda)
    newWeightMatrix = computeWeight(DC_points,numLayers,Depth)
    newPhase = Forward(newWeightMatrix,initialDataProfile[:,0])
    # ax[1].plot(newPhase,newLayerSet[1:],'-ro',markerfacecolor='None')
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Phase velocity, Vph [m/s]')
    ax[1].set_ylabel('Wavelength, [m]')
    ax[1].xaxis.set_label_position('top')
    ax[1].xaxis.tick_top()
    ax[1].spines['bottom'].set_color('white')
    ax[1].spines['right'].set_color('white')
    # ax.set_xlim(0,Vs[-1])
    # ax.spines['bottom'].set_color('white')
    # ax.spines['right'].set_color('white')
    ax2 = ax[1].twinx()
    Vs = np.append(Vs,Vs[-1])
    Depth[-1] = Depth[-2] + 2
    Depth = np.append(0,Depth)
    ax2.step(Vs,Depth[1:],'r',label='Initialized Vs-profile')
    ax2.invert_yaxis()
    ax2.set_xlabel('Shear velocity, Vph [m/s]')
    ax2.set_ylabel('Depth, [m]',color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax[1].legend(loc='upper right')
    ax2.legend(loc='lower left')
    plt.show()
plot()
