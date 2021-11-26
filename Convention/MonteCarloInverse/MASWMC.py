import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ttictoc import tic,toc
from iteration_utilities import all_monotone
##
FigFontSize = 11
resolution = 100
##
Filename = 'SampleData_MASWavesInversion.csv'
# sample_data = pd.read_csv(Filename,encoding="utf-8")
sample_data = np.genfromtxt(Filename,delimiter=',',dtype=None)
print(sample_data)
c_OBS = sample_data[:,0]
print(c_OBS)
up_low_boundary = 'Yes'
c_OBS_low = sample_data[:,1]
c_OBS_up = sample_data[:,2]
lambda_OBS = sample_data[:,3]
print(lambda_OBS)
c_min, c_max = 75, 300
c_step = 0.09
delta_c = 2
n = 4
h_initial = [2,2,4,6]
beta_initial = [1.09*c_OBS[0], 1.09*c_OBS[9], 1.09*c_OBS[16], 1.09*c_OBS[22], 1.09*c_OBS[-1]]
n_unsat = 1
nu_unsat = 0.3
alpha_temp = [np.sqrt(2*(1-nu_unsat)/(1-2*nu_unsat)) * beta_initial[i] for i in range(len(beta_initial))]
alpha_initial = [alpha_temp[0], 1440, 1440, 1440, 1440]
rho = [1850, 1900, 1950, 1950, 1950]

N_reversals = 0 # Normally dispersive analysis
# Search control parameters
b_S = 5
b_h = 10
N_max = 1000
e_max = 0

class Monte_Carlo_Inverse():
    def __init__(self,c_min,c_max,c_step,delta_c,n,n_unsat,alpha_initial,nu_unsat,beta_initial,
                rho,h_initial,N_reversals,c_OBS,lambda_OBS,up_low_boundary,c_OBS_up,c_OBS_low,
                b_S,b_h,N_max,e_max):
        self.c_min = c_min
        self.c_max = c_max
        self.c_step = c_step
        self.delta_c = delta_c
        self.n = n
        self.n_unsat = n_unsat
        self.alpha_initial = alpha_initial
        self.nu_unsat = nu_unsat
        self.beta_initial = beta_initial
        self.rho = rho
        self.h_initial = h_initial
        self.N_reversals = N_reversals
        self.c_OBS = c_OBS
        self.lambda_OBS = lambda_OBS
        self.up_low_boundary = up_low_boundary
        self.c_OBS_up = c_OBS_up
        self.c_OBS_low = c_OBS_low
        self.b_S = b_S
        self.b_h = b_h
        self.N_max = N_max
        self.e_max = e_max

    def Misfit(self,c_t):
        Q = len(c_t)
        e = (1/Q) * sum(np.sqrt(self.c_OBS - c_t)**2)*100
        return e

    def FWMA_recursion(self,c,k,alpha,beta1,beta2,rho1,rho2,h,X):
        gamma1 = beta1**2/c**2
        gamma2 = beta2**2/c**2
        epsilon = rho2/rho1
        eta = 2*(gamma1-epsilon*gamma2)
        a = epsilon+eta
        ak = a-1
        b = 1-eta
        bk = b-1

        # Layer eigenfunctions
        r = np.sqrt(1-(c**2/alpha**2))
        C_alpha = np.cosh(k*r*h)
        S_alpha = np.sinh(k*r*h)

        s = np.sqrt(1-(c**2/beta1**2))
        C_beta = np.cosh(k*s*h)
        S_beta = np.sinh(k*s*h)

        # Layer recursion
        p1 = C_beta*X[2]+s*S_beta*X[3]
        p2 = C_beta*X[3]+s*S_beta*X[4]
        p3 = (1/s)*S_beta*X[2]+C_beta*X[3]
        p4 = (1/s)*S_beta*X[3]+C_beta*X[4]

        q1 = C_alpha*p1-r*S_alpha*p2
        q2 = -(1/r)*S_alpha*p3+C_alpha*p4
        q3 = C_alpha*p3-r*S_alpha*p4
        q4 = -(1/r)*S_alpha*p1+C_alpha*p2

        y1 = ak*X[1]+a*q1
        y2 = a*X[1]+ak*q2

        z1 = b*X[1]+bk*q1
        z2 = bk*X[1]+b*q2

        X[0] = bk*y1+b*y2
        X[1] = a*y1+ak*y2
        X[2] = epsilon*q3
        X[3] = epsilon*q4
        X[4] = bk*z1+b*z2
        return X

    def MASW_fast_delta_matrix_analysis(self,c,k,n,alpha,beta,rho,h):

        # initialization
        X_temp = np.array([2*(2-c**2/beta[0]**2), -(2-c**2/beta[0]**2)**2, 0, 0, -4])
        X = [(rho[0]*beta[0]**2)**2 * X_temp[i] for i in range(len(X_temp))]
        # layer recursion
        for i in range(self.n):
            X = self.FWMA_recursion(c,k,alpha[i],beta[i],beta[i+1],rho[i],rho[i+1],h[i],X)
        # dispersion function value
        r = np.sqrt(1-c**2/alpha[n]**2)
        s = np.sqrt(1-c**2/beta[n]**2)
        D = (X[1] + X[2]) - r * (X[3] + s*X[4])
        return D

    def MASW_theorectical_dispersion(self):
        c_test = np.arange(self.c_min,self.c_max+self.c_step,c_step)
        k = (2*np.pi) / self.lambda_OBS
        modes = 1
        D = np.zeros((len(k),len(c_test)))
        c_t = np.zeros((len(k),modes))
        lambda_t = np.zeros((len(k),modes))
        m_loc = 0
        delta_m = round(self.delta_c/self.c_step)

        for j in range(len(k)):
            for m in range(m_loc,len(c_test)):
                D[j,m] = self.MASW_fast_delta_matrix_analysis(c_test[m],k[j],self.n,self.alpha_initial,
                self.beta_initial,self.rho,self.h_initial)
                if m == m_loc:
                    sign_old = np.sign(self.MASW_fast_delta_matrix_analysis(c_test[0],k[j],self.n,self.alpha_initial,
                    self.beta_initial,self.rho,self.h_initial))
                else:
                    sign_old = signD
                signD = np.sign(D[j,m])
                if sign_old * signD == -1:
                    c_t[j] = c_test[m]
                    lambda_t[j] = 2*np.pi/k[j]
                    m_loc = m - delta_m
                    if m_loc <= 0:
                        m_loc = 1
                        break
        return c_t, lambda_t

    def do_inversion(self):
        c_t,lambda_t = self.MASW_theorectical_dispersion()
        e_initial = self.Misfit(c_t)

        '''
        timerVal = tic
        store_all = np.zeros((6,self.N_max),dtype = 'object')
        beta_opt = self.beta_initial
        h_opt = self.h_initial
        e_opt = e_initial
        Nend = self.N_max

        for w in range(self.N_max):
            # [w]
            beta_test = beta_opt + np.random.uniform(-(self.b_S/100)*beta_opt,self.b_S/100*beta_opt)
            while all_monotone(beta_test[self.reversals+1:]) == False:
                beta_test = beta_opt + np.random.uniform(-(self.b_S/100)*beta_opt,self.b_S/100*beta_opt)
            h_test = h_opt + np.random.uniform(-(self.b_h/100)*h_opt,self.b_h/100*h_opt)
        '''
        return c_t,lambda_t,e_initial

MC = Monte_Carlo_Inverse(c_min,c_max,c_step,delta_c,n,n_unsat,alpha_initial,nu_unsat,beta_initial,
                rho,h_initial,N_reversals,c_OBS,lambda_OBS,up_low_boundary,c_OBS_up,c_OBS_low,
                b_S,b_h,N_max,e_max)
a = MC.do_inversion()
print(a)