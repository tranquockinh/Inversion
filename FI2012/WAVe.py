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
waveLength0 = 2
deltaWaveLength = 2
Lambda = np.arange(waveLength0,45,deltaWaveLength)
def dataInput(soilProfile):
    numLayers = len(soilProfile[:,0])
    Depth =np.append(0, soilProfile[:,1])
    Vs = soilProfile[:,0]
    return numLayers,Depth,Vs
numLayers,Depth,Vs = dataInput(Layer_Data)
# Compute weighting factor using weighting function

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

def weighting2011(D,WL):
	z = sp.symbols('z')
	limit_low = D[:-1]
	limit_up = D[1:]
	material_coefficient = 1
	cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi,-0.3933*2*np.pi])
	cv1,cv2,cv3,cv4 = cv[0],cv[1],cv[2],cv[3]
	PDF = (cv1*sp.exp(cv3/WL*z) + cv2*sp.exp(cv4/WL*z))*material_coefficient
	num_layer = len(D) - 1
	layerArea2011 = np.zeros((num_layer),dtype='object')
	A_term1 = (cv1/(cv3/WL))*(sp.exp(cv3/WL*np.inf) - sp.exp(cv3/WL*0))
	A_term2 = (cv2/(cv4/WL))*(sp.exp(cv4/WL*np.inf) - sp.exp(cv4/WL*0))
	Area = A_term1 + A_term2
	for j in range(num_layer):
		#Area_i[j] = sp.integrate(PDF,(z,limit_low[j],limit_up[j]))
		term1 = (cv1/(cv3/WL))*(sp.exp(cv3/WL*limit_up[j]) - sp.exp(cv3/WL*limit_low[j]))
		term2 = (cv2/(cv4/WL))*(sp.exp(cv4/WL*limit_up[j]) - sp.exp(cv4/WL*limit_low[j]))
		layerArea2011[j] = term1 + term2
	weights2011 = layerArea2011 / Area
	return layerArea2011,weights2011

def weighting(Depth,Wavelength):
    numLayers = len(Depth) - 1
    z = sp.symbols('z')
    limit_low = Depth[:-1]
    limit_up = Depth[1:]
    integralPDF = z - (2/5) * ((z/Wavelength)**(5/2)) * Wavelength
    layerArea = np.zeros((numLayers),dtype='object')
    Area = integralPDF.subs({z:Wavelength}) - integralPDF.subs({z:0})
    for i in range(numLayers):
        if limit_up[i] <= Wavelength:
            layerArea[i] = integralPDF.subs({z:limit_up[i]}) - integralPDF.subs({z:limit_low[i]})
        else:
            layerArea[i] = integralPDF.subs({z:Wavelength}) - integralPDF.subs({z:limit_low[i]})
            break
    weights = layerArea/Area
    return layerArea,weights

def computeWeight(numLayers,Depth,waveLength0,deltaWaveLength):
    # Lambda = np.arange(waveLength0,Depth[-2]+2*deltaWaveLength,deltaWaveLength)
    DC_points = len(Lambda)
    WEIGHTS = np.zeros((DC_points,numLayers))
    WEIGHTS2011 = np.zeros((DC_points,numLayers))
    for i in range(DC_points):
        layerArea,weights = weighting(Depth,Lambda[i])
        layerArea,weights2011 = weighting2011(Depth,Lambda[i])
        WEIGHTS[i,:] = weights
        WEIGHTS2011[i,:] = weights2011
    print('\n\nThe Weight matrix computed from example Vs profile and an array of wavelength:\n')
    plotTable(WEIGHTS,index ='Lambda {}',columns='Layer {}',Name = 'Weight Matrix')
    return WEIGHTS,WEIGHTS2011
WEIGHTS,WEIGHTS2011 = computeWeight(numLayers,Depth,waveLength0,deltaWaveLength)
Vph = 0.93*np.matmul(WEIGHTS,Vs)
Vph2011 = 0.93*np.matmul(WEIGHTS2011,Vs)

def plotParticleDispl(refineDepth,atWavelength):
    layerArea,fineWeight = weighting(refineDepth,atWavelength)
    layerArea2011,fineWeight2011 = weighting2011(refineDepth,atWavelength)
    plotData = np.zeros((len(layerArea),2))
    plotData2011 = np.zeros((len(layerArea),2))
    for i in range(len(layerArea)):
        plotData[:,0] = refineDepth[1:]
        plotData[:,1] = fineWeight/fineWeight[0]
        plotData2011[:,0] = refineDepth[1:]
        plotData2011[:,1] = fineWeight2011/fineWeight2011[0]
    plotTable(plotData,index ='Refine depth {}',columns='Displ. {}',Name ='Displ. distribution')
    fig,ax = plt.subplots(figsize = (4,6),dpi=100)
    ax.plot(plotData[:,1],plotData[:,0],label='Leong, 2012')
    ax.plot(plotData2011[:,1],plotData2011[:,0],label='Cao, 2011')
    return fig,ax

refineDepth = np.arange(0,10.0,0.5)
Depth[-1] = np.inf
atWavelength = 5

fig,ax = plotParticleDispl(refineDepth,atWavelength)
ax.invert_yaxis()
ax.set_xlabel('Normalized vertical displacement')
ax.set_ylabel('depth, (m)')
ax.legend()
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
plt.show()


fig,ax = plt.subplots(figsize=(4,6),dpi=100)
ax.plot(Vph,Lambda,'-bo',markerfacecolor='None',label='Leong, 2012')
ax.plot(Vph2011,Lambda,'-r*',label='Cao, 2011')
ax.invert_yaxis()
ax.set_xlabel('Shear wave velocity (m/s)')
ax.set_ylabel('Wavelength, (m)')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
ax.legend()
plt.show()
