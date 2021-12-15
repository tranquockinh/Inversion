#%%
import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *
#%%
shearWave = [150,300,400]
thickness = [5,10,np.inf]
lambda_min,lambda_max, delta_lambda = 3,44,6
wl = np.arange(lambda_min,lambda_max+delta_lambda,delta_lambda)
displ_function = input('input 1 for Cao,2011; 2 if Leong,2012 version: ')
#%%
class forward():
    def __init__(self,shearWave,thickness):
        self.t = thickness
        self.t = np.append(0,self.t)
        self.Swave = shearWave
        self.nLayer = len(self.t)-1

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
        W = np.zeros((len(wl),self.nLayer))
        for i,lambdaValue in enumerate(wl):
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
        Swave = np.dot(W,self.Swave)
        return Swave,Rwave

    def refineLayer(self,numLayer,coeff):
        layers = [0]
        for i in range(len(wl[:numLayer])):
            if i == (len(wl[:numLayer])-1):
                layers.append(np.inf)
            else:
                layers.append(wl[:numLayer][i]*coeff)
        return layers

fw = forward(shearWave,thickness)
coeff = 0.3
test_nLayer = len(wl)
test_thickness = fw.refineLayer(test_nLayer,coeff)[1:]
test_Swave,test_Rwave = fw.DC()
W,invertW = fw.weightMatrix()
#%%
class backward():
    def __init__(self):
        self.DC = forward(test_Swave[:test_nLayer],test_thickness)
    
    def inversionData(self):
        W,invertW = self.DC.weightMatrix()
        Swave,Rwave = self.DC.DC()
        return W,invertW,Swave,Rwave

    def inversion(self):
        _,invertW,Swave,Rwave = self.inversionData()
        iSwave = np.matmul(invertW,Swave)
        return iSwave

bw = backward()
W,invertW,Swave,Rwave = bw.inversionData()
iSwave = bw.inversion()
print(iSwave)
    