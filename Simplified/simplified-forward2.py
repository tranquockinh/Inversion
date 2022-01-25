#%%
import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *
#%%
shearWave = [150,300,400]
thickness = [5,10,np.inf]
lambda_min,lambda_max, delta_lambda = 6,40,6
wl = np.arange(lambda_min,lambda_max+delta_lambda,delta_lambda)
displ_function = input('input 1 for Cao,2011; 2 if Leong,2012 version: ')
#%%
class forward():
    def __init__(self,wl,shearWave,thickness):
        self.t = thickness
        self.t = np.append(0,self.t)
        self.Swave = shearWave
        self.nLayer = len(self.t)-1
        self.Lambda = wl
    def boundaries(self):
        z = sp.Symbol('z')
        low = self.t[:-1]
        up = self.t[1:]
        return z,up,low

    def pdf1(self,lambdaScalar):
        self.lam = lambdaScalar
        z,up,low = self.boundaries()
        ipdf = z-0.25*(z/self.lam)**(2.5)*self.lam
        unit = np.zeros((self.nLayer))
        totA = ipdf.subs({z:self.lam})-ipdf.subs({z:0})
        for j in range(self.nLayer):
            if up[j] <= self.lam:
                unit[j] = ipdf.subs({z:up[j]})-ipdf.subs({z:low[j]})
            else:
                unit[j] = ipdf.subs({z:self.lam})-ipdf.subs({z:low[j]})
                break
        weights = unit/totA
        return weights

    def pdf2(self,lambdaScalar):
        self.lam = lambdaScalar
        z,up,low = self.boundaries()
        cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi, -0.3933*2*np.pi])
        totA_term1 = (cv[0]/(cv[2]/self.lam))*(sp.exp(cv[2]/self.lam*np.inf)-sp.exp(cv[2]/self.lam*0))
        totA_term2 = (cv[1]/(cv[3]/self.lam))*(sp.exp(cv[3]/self.lam*np.inf)-sp.exp(cv[3]/self.lam*0))
        totA = totA_term1 + totA_term2
        unit = np.zeros((self.nLayer))
        for j in range(self.nLayer):
            unit_term1 = (cv[0]/(cv[2]/self.lam))*(sp.exp(cv[2]/self.lam*up[j])-sp.exp(cv[2]/self.lam*low[j]))
            unit_term2 = (cv[1]/(cv[3]/self.lam))*(sp.exp(cv[3]/self.lam*up[j])-sp.exp(cv[3]/self.lam*low[j]))
            unit[j] = unit_term1 + unit_term2
        weights = unit/totA
        return weights

    def weightMatrix(self):
        W = np.zeros((len(self.Lambda),self.nLayer))
        for i,lambdaValue in enumerate(self.Lambda):
            if displ_function == 1:
                weights = self.pdf1(lambdaValue)
            else:
                weights = self.pdf2(lambdaValue)
            W[i,:] = weights
        U,S,VT = np.linalg.svd(W,full_matrices=False)
        invertW = np.dot(VT.T,np.dot(np.linalg.inv(np.diag(S)),U.T))
        return W,invertW

    def DC(self):
        W,_ = self.weightMatrix()
        Swave = np.dot(W,self.Swave)
        Rwave = 0.93*Swave
        return Swave,Rwave

    def refineLayer(self,numLayer,coeff):
        layers = [0]
        for i in range(len(wl[:numLayer])):
            if i == (len(wl[:numLayer])-1):
                layers.append(np.inf)
            else:
                layers.append(wl[:numLayer][i]*coeff)
        return layers

#%%
class backward():
    def __init__(self,wl,shearWave,thickness,coeff,test_nLayer):
        self.fw = forward(wl,shearWave,thickness)
        self.test_nLayer = test_nLayer
        fullSwave,fullRwave = self.fw.DC()
        fullThickness = self.fw.refineLayer(len(wl),coeff)[1:]
        self.coeff = coeff
        self.wl_test = wl[:test_nLayer]
        self.Swave_test = fullSwave[:test_nLayer]
        self.Rwave_test = fullRwave[:test_nLayer]
        self.thick_test = fullThickness[:test_nLayer]
        self.thick_test[-1] = np.inf

    def initialization(self):
        ini_weight = np.zeros((self.test_nLayer,self.test_nLayer))
        for i in range(self.test_nLayer):
            thickness = self.fw.refineLayer(i+1,self.coeff)[1:]
            ini_fw = forward(wl[:i+1],shearWave,thickness)
            W,invertW = ini_fw.weightMatrix()
            ini_weight[:i+1,:i+1] = W[::1,::1]
        self.init_Vs = np.matmul(ini_weight,self.Swave_test)
        return self.init_Vs,thickness

    def inversion(self):
        fw2 = forward(self.wl_test,self.init_Vs,self.thick_test)
        W,invertW = fw2.weightMatrix()
        self.iSwave = np.matmul(invertW,self.Swave_test)
        return self.iSwave

    def check_forward(self):
        checkFD = forward(wl,self.iSwave,self.thick_test)
        checkSwave,checkRwave = checkFD.DC()
        return checkSwave,checkRwave
    
    def parameters(self):
        test_Rwave = self.Rwave_test
        return test_Rwave

bw = backward(wl,shearWave,thickness,1/3,len(wl))

#%% ploting data
_,invthickness = bw.initialization()
inverted_Vs = bw.inversion()
checkSwave,checkRwave = bw.check_forward()
test_Rwave = bw.parameters()
iniSwave = shearWave
iniThickness = thickness
invSwave = inverted_Vs


def plot(iniSwave,iniThickness,invSwave,invThickness,test_Rwave,checkRwave,wl):
    
    
    iniThickness[-1] = iniThickness[-2] + 0.5*iniThickness[-2]
    invThickness[-1] = iniThickness[-1]
    iniThickness = np.append(0,iniThickness)
    invThickness = np.append(0,invThickness)
    iniSwave = np.append(iniSwave,iniSwave[-1])
    invSwave = np.append(invSwave,invSwave[-1])

    fig,ax = plt.subplots(1,2,figsize=(10,6))
    ax[0].step(iniSwave,iniThickness,'-r',linewidth=2,label='True')
    ax[0].step(invSwave,invThickness,'--b',label='Inverted')
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Shear velocity, Vph [m/s]')
    ax[0].set_ylabel('Depth, [m]')
    ax[0].xaxis.set_label_position('top')
    ax[0].xaxis.tick_top()
    ax[0].spines['bottom'].set_color('white')
    ax[0].spines['right'].set_color('white')
    ax[0].legend()

    ax[1].plot(test_Rwave,wl,'-or',linewidth=2,markerfacecolor='r',label='Inverted')
    ax[1].plot(checkRwave,wl,'-ob',linewidth=2,markerfacecolor='None',label='Check')
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Phase velocity, Vph [m/s]')
    ax[1].set_ylabel('Wavelength, [m]')
    ax[1].xaxis.set_label_position('top')
    ax[1].xaxis.tick_top()
    ax[1].spines['bottom'].set_color('white')
    ax[1].spines['right'].set_color('white')
    ax[1].legend()
    plt.show()

plot(iniSwave,iniThickness,invSwave,invthickness,test_Rwave,checkRwave,wl)
