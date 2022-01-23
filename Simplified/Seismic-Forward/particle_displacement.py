from dis import dis
import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *
# shear wave velocity
Swave = [150,400]
# thickness
thickness = [3,np.inf]
depth = np.append(0,thickness)
nLayer = len(thickness)
# Poisson's ratio
nuy = 0.3
# shear- to compression-wave coefficinet
spr = 0.93
# depth to
# wavelength ratio
alpha = 4/5
# number of layer inverted
n_inverted_layer = 3 #len(wavelen)
# import forward2

# fw = forward2.forward_engine(depth,Swave,wavelen=np.array([5]))
# w = fw.weight_allLambda()
# wa = fw.weight_atLambda(2)
# print(wa)

class forward_engine(object):
    
    def __init__(self,depth):
        self.depth = depth
        self.nunber_layers = len(depth) - 1

    def boundaries(self):
        z = sp.Symbol('z')
        low = self.depth[:-1]
        up = self.depth[1:]
        return z,up,low

    def pdf1(self,lambdaScalar):
        self.lam = lambdaScalar
        z,up,low = self.boundaries()
        ipdf = z-0.25*(z/self.lam)**(2.5)*self.lam
        unit = np.zeros((self.nunber_layers))
        totA = ipdf.subs({z:self.lam})-ipdf.subs({z:0})
        for j in range(self.nunber_layers):
            if up[j] <= self.lam:
                unit[j] = ipdf.subs({z:up[j]})-ipdf.subs({z:low[j]})
            else:
                unit[j] = ipdf.subs({z:self.lam})-ipdf.subs({z:low[j]})
                break
        weights = unit/totA
        return unit,weights

    def pdf2(self,lambdaScalar):
        self.lam = lambdaScalar
        z,up,low = self.boundaries()
        cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi, -0.3933*2*np.pi])
        totA_term1 = (cv[0]/(cv[2]/self.lam))*(sp.exp(cv[2]/self.lam*np.inf)-sp.exp(cv[2]/self.lam*0))
        totA_term2 = (cv[1]/(cv[3]/self.lam))*(sp.exp(cv[3]/self.lam*np.inf)-sp.exp(cv[3]/self.lam*0))
        totA = totA_term1 + totA_term2
        unit = np.zeros((self.nunber_layers))
        for j in range(self.nunber_layers):
            unit_term1 = (cv[0]/(cv[2]/self.lam))*(sp.exp(cv[2]/self.lam*up[j])-sp.exp(cv[2]/self.lam*low[j]))
            unit_term2 = (cv[1]/(cv[3]/self.lam))*(sp.exp(cv[3]/self.lam*up[j])-sp.exp(cv[3]/self.lam*low[j]))
            unit[j] = unit_term1 + unit_term2
        weights = unit/totA
        return unit,weights

test_particle_displacement = forward_engine(depth)
disp, weights = test_particle_displacement.pdf1(lambdaScalar=6)
disp2, weights2 = test_particle_displacement.pdf2(lambdaScalar=6)
print(disp2)