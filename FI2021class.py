import numpy as np
import sympy as sp
from sympy.core.symbol import disambiguate
from sympy.functions.elementary.exponential import LambertW
import tkinter
import matplotlib.pyplot as plt
from pandas import *

shearWave = [100,200]
thickness = [5,np.inf]
lambda_min,lambda_max, delta_lambda = 6,40,2

lambdaVec = np.arange(lambda_min,lambda_max+delta_lambda,delta_lambda)
class forward(object):
    def __init__(self,shearWave,thickness,lambdaVec):
        self.sWave = shearWave
        self.t = thickness
        self.lamVec = lambdaVec
        self.t = np.append(0,self.t)
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
            multiplier_1 = (sp.exp(cv[2]/self.lam*up[j])-sp.exp(cv[3]/self.lam*low[j]))
            unit_term1 = (cv[0]/(cv[2]/self.lam))*multiplier_1
            multiplier_2 = (sp.exp(cv[3]/self.lam*up[j])-sp.exp(cv[3]/self.lam*low[j]))
            unit_term2 = (cv[1]/(cv[3]/self.lam))*multiplier_2
            unit[j] = unit_term1 + unit_term2
        weights = unit/totA
        return weights
    def weightMat(self):
        W = np.zeros((len(self.lamVec),self.nLayer))
        for i,lambdaValue in enumerate(self.lamVec):
            weights = self.pdf1(lambdaValue)
            W[i,:] = weights
        return W
    def svd(self):
        W = self.weightMat()
        U,S,VT = np.linalg.svd(W,full_matrices=False)
        get_W = np.dot(U,np.dot(np.linalg.inv(np.diag(S)),VT))
        invert_svd_W = np.dot(VT.T,np.dot(np.linalg.inv(np.diag(S)),U.T))
        return get_W,invert_svd_W
    def dispersionCurve(self):
        waveMatrix = self.weightMat()
        rWave = 0.93*np.dot(waveMatrix,self.sWave)
        return rWave
    def newLayer(self,input_nLayer,coeff):
        newLayers = [0]
        for i in range(len(self.lamVec[:input_nLayer])):
            if i == (len(self.lamVec[:input_nLayer])-1):
                newLayers.append(np.inf)
            else:
                newLayers.append(self.lamVec[:input_nLayer][i]*coeff)
        return newLayers

class backward(object):
    def __init__(self):
        self.dispersion = forward(shearWave,thickness,lambdaVec)
    def weight_table(self,Matrix,index ='Lambda {}',columns='Layer {}',Name = ''):
        rowLabels,colLabels = [],[]
        for i in range(len(Matrix[:,0])):
            rowLabels.append(index.format(i+1))
        for j in range(len(Matrix[0,:])):
            colLabels.append(columns.format(j+1))
        weights_new_df = DataFrame(Matrix,index=Index(rowLabels,name=Name),columns=colLabels)
        return weights_new_df
    def inversionData(self,input_nLayer,lam_depth_coeff):
        input_thickness = self.dispersion.newLayer(input_nLayer,lam_depth_coeff)[1:]
        input_rWave = self.dispersion.dispersionCurve()
        input_sWave_convert = (1/0.93)*input_rWave
        input_lamda_Vec = lambdaVec
        return input_sWave_convert,input_thickness,input_lamda_Vec
    def new_weight_matrix_complete(self,lam_depth_coeff):
        input_sWave_convert,input_thickness,input_lamda_Vec = self.inversionData(len(self.dispersion.weightMat()),lam_depth_coeff)
        dispersion_new = forward(input_sWave_convert,input_thickness,input_lamda_Vec)
        weights_new = dispersion_new.weightMat()
        weights_new_table = self.weight_table(weights_new)
        return weights_new_table
    def new_weight_matrix_checkpoints(self,input_nPoints,lam_depth_coeff):
        input_sWave_convert,input_thickness,input_lamda_Vec = self.inversionData(input_nPoints,lam_depth_coeff)
        input_sWave_convert_check = input_sWave_convert[:input_nPoints]
        input_thickness_check = input_thickness[:input_nPoints]
        input_lamda_Vec_check = input_lamda_Vec[:input_nPoints]
        dispersion_new_check = forward(input_sWave_convert_check,input_thickness_check,input_lamda_Vec_check)
        weight_new_check = dispersion_new_check.weightMat()
        weight_new_check_table = self.weight_table(weight_new_check)
        return weight_new_check_table


lambda_to_z_ratio = 1
check_nLayer = 2

dispersion = forward(shearWave,thickness,lambdaVec)
print(dispersion.dispersionCurve())
test = backward()
print(test.new_weight_matrix_complete(1))
print(test.new_weight_matrix_checkpoints(check_nLayer,lambda_to_z_ratio))
