import numpy as np
from numpy.lib.nanfunctions import nanmedian
import sympy as sp
from sympy.core.symbol import disambiguate
from sympy.functions.elementary.exponential import LambertW
import tkinter
import matplotlib.pyplot as plt
from pandas import *

shearWave = [150,200,250,400]
thickness = [2,5,10,np.inf]
lambda_min,lambda_max, delta_lambda = 6,40,2
total_refine_layers = len(np.arange(lambda_min,lambda_max+delta_lambda,delta_lambda))
name_displ_function = input('inpu 1 if use (Cao,2011) version and input 2 if use (Leong, 2012) version: ')
lambdaVec = np.arange(lambda_min,lambda_max+delta_lambda,delta_lambda)

class forward(object):
    def __init__(self,shearWave,thickness,lambdaVec,name_displ_function):
        self.sWave = shearWave
        self.t = thickness
        self.lamVec = lambdaVec
        self.t = np.append(0,self.t)
        self.nLayer = len(self.t)-1
        self.par_disp_func = name_displ_function

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

    def weightMat(self):
        W = np.zeros((len(self.lamVec),self.nLayer))
        for i,lambdaValue in enumerate(self.lamVec):
            if self.par_disp_func == 1:
                weights = self.pdf1(lambdaValue)
            else:
                weights = self.pdf2(lambdaValue)
            W[i,:] = weights
        return W

    def svd(self):
        W = self.weightMat()
        U,S,VT = np.linalg.svd(W,full_matrices=False)
        get_W = np.dot(U,np.dot(np.linalg.inv(np.diag(S)),VT))
        invert_svd_W = np.dot(VT.T,np.dot(np.linalg.inv(np.diag(S)),U.T))
        return invert_svd_W

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
        print(newLayers)
        return newLayers

class backward(object):
    def __init__(self,check_nLayer,wavelength_depth_rat):
        self.input_nLayer = check_nLayer
        self.lam_depth_coeff = wavelength_depth_rat
        self.dispersion = forward(shearWave,thickness,lambdaVec,name_displ_function)

    def weight_table(self,Matrix,index ='Lambda {}',columns='Layer {}',Name = ''):
        rowLabels,colLabels = [],[]
        for i in range(len(Matrix[:,0])):
            rowLabels.append(index.format(i+1))
        for j in range(len(Matrix[0,:])):
            colLabels.append(columns.format(j+1))
        weights_new_df = DataFrame(Matrix,index=Index(rowLabels,name=Name),columns=colLabels)
        return weights_new_df

    def inversionData(self):
        input_thickness = self.dispersion.newLayer(self.input_nLayer,self.lam_depth_coeff)[1:]
        input_rWave = self.dispersion.dispersionCurve()
        input_sWave_convert = (1/0.93)*input_rWave
        input_lamda_Vec = lambdaVec
        return input_rWave,input_sWave_convert,input_thickness,input_lamda_Vec

    def new_weight_matrix_checkpoints(self):
        _,input_sWave_convert,input_thickness,input_lamda_Vec = self.inversionData()
        self.input_sWave_convert_check = input_sWave_convert[:self.input_nLayer]
        input_thickness_check = input_thickness[:self.input_nLayer]
        input_lamda_Vec_check = input_lamda_Vec[:self.input_nLayer]
        self.dispersion_new_check = forward(self.input_sWave_convert_check,input_thickness_check,input_lamda_Vec_check,name_displ_function)
        weight_new_check = self.dispersion_new_check.weightMat()
        new_phase_velocity = self.dispersion_new_check.dispersionCurve()
        weight_new_check_table = self.weight_table(weight_new_check)
        return weight_new_check_table,new_phase_velocity

    def inversion(self):
        invert_svd_W_check = self.dispersion_new_check.svd()
        convert_sWave = np.matmul(invert_svd_W_check,self.input_sWave_convert_check)
        return convert_sWave
    
    def check_dispersion_data(self):
        match_shear_wave = self.dispersion_new_check.dispersionCurve()
        return match_shear_wave

wavelength_depth_rat = 1
check_nLayer = total_refine_layers

dispersion = forward(shearWave,thickness,lambdaVec,name_displ_function)
test = backward(check_nLayer,wavelength_depth_rat)

new_weight,new_phase_velocity = test.new_weight_matrix_checkpoints()
convert_Swave = test.inversion()

def plot(Lambda,Vph,new_phase_velocity):
    fig,ax = plt.subplots(figsize=(4,5),dpi=100)
    ax.plot(Vph,Lambda,'-bo',markerfacecolor='None')
    ax.plot(new_phase_velocity,Lambda[:len(new_phase_velocity)],'-ro',markerfacecolor='None')
    ax.invert_yaxis()
    ax.set_xlabel('Phase velocity, Vph [m/s]')
    ax.set_ylabel('Wavelength, [m]')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')

    plt.show()

Phase_velocity,Shear_velocity,Thickness,Wavelegnth = test.inversionData()
plot(Wavelegnth,Phase_velocity,new_phase_velocity)