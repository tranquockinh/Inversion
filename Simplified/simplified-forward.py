#%%
import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *
#%%
shearWave = [150,300,400]
thickness = [5,10,np.inf]
lambda_min,lambda_max, delta_lambda = 5,44,6
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

fw = forward(wl,shearWave,thickness)
coeff = 1/3
fullSwave,fullRwave = fw.DC()
fullThickness = fw.refineLayer(len(wl),coeff)[1:]
test_nLayer = len(wl)

wl_test = wl[:test_nLayer]
Swave_test = fullSwave[:test_nLayer]
thick_test = fullThickness[:test_nLayer]
thick_test[-1] = np.inf

# fw = forward(wl_test,Swave_test,thick_test)
# test_Swave,test_Rwave = fw.DC()
# test_thickness = fw.refineLayer(test_nLayer,coeff)[1:]
W,invertW = fw.weightMatrix()

#%%
class backward():
    def __init__(self):
        self.DC = forward(wl[:test_nLayer],Swave_test,thick_test)
    
    def inversionData(self):
        W,invertW = self.DC.weightMatrix()
        Swave,Rwave = self.DC.DC()
        return W,invertW,Swave,Rwave

    def inversion(self):
        _,invertW,Swave,_ = self.inversionData()
        self.iSwave = np.matmul(invertW,Swave)
        return self.iSwave
    
    def check_forward(self):
        checkFD = forward(wl[:test_nLayer],self.iSwave,thick_test)
        # check_W,_ = checkFD.weightMatrix()
        # checkRwave = 0.93 * np.matmul(check_W,self.iSwave)
        checkSwave,checkRwave = checkFD.DC()
        return checkSwave,checkRwave

bw = backward()
W,invertW,Swave,Rwave = bw.inversionData()
iSwave = bw.inversion()
checkSwave,checkRwave = bw.check_forward()


def plot(iniSwave,iniThickness,invSwave,invThickness,test_Rwave,test_thickness):
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
    ax[1].plot(test_Rwave,test_thickness,'-or',linewidth=2,markerfacecolor='None')
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Phase velocity, Vph [m/s]')
    ax[1].set_ylabel('Wavelength, [m]')
    ax[1].xaxis.set_label_position('top')
    ax[1].xaxis.tick_top()
    ax[1].spines['bottom'].set_color('white')
    ax[1].spines['right'].set_color('white')
    plt.show()

plot(shearWave,thickness,iSwave,thick_test,fullRwave,fullThickness)

