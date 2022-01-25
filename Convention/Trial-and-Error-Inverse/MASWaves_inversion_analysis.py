#%%
import numpy as np
import matplotlib.pyplot as plt
##
FigFontSize = 14
##

c_test_min = 0
c_test_max = 500
delta_c_test = 0.5
c_test = np.arange(c_test_min,c_test_max+delta_c_test,delta_c_test)
n = 6
alpha = np.array([1400,1400,1400,1400,1400,1400,1400])
h = np.array([1,1,2,2,4,5,np.inf])
beta = np.array([75,90,150,180,240,290,290])
# beta = np.array([75,90,150,180,240,290,290])
# beta = np.array([100,120,150,150,240,280,280])
rho = [1850,1850,1850,1850,1850,1850,1850]
plot_exp_the_DC = 'Yes'
dispersionData = np.loadtxt('dispersion_data.txt', dtype='float', delimiter=None)
c_curve0,lambda_curve0 = dispersionData[:,0],dispersionData[:,1]
h_plot = np.zeros((len(h)))
for i in range(len(h)):
    if i == 0:
        h_plot[i] = h[i]
    else:
        h_plot[i] = h_plot[i-1] + h[i]
h_plot[-1] = h_plot[-2] + 4
h_plot[-1] += 4
beta_plot = beta

beta_plot = np.append(beta_plot[0],beta_plot)
h_plot = np.append(0,h_plot)

plt.figure(figsize=(4,6))
plt.step(beta_plot,h_plot)
plt.xlabel('Shear wave velocity [m/s]',fontsize = FigFontSize)
plt.ylabel('Depth [m]', fontsize = FigFontSize)
plt.gca().invert_yaxis()
#%%
class seismic_backward(object):
    def __init__(self,n,beta,h,alpha,rho,c_test,c_curve0,lambda_curve0):
        self.n = n
        self.alpha = alpha
        self.h = h
        self.beta = beta
        self.rho = rho
        self.c_test = c_test
        self.c_exp = c_curve0
        self.lambda_exp = lambda_curve0
        self.wavenumber = (2*np.pi)/(self.lambda_exp)

    def layer_stiffness_matrix(self,h,alpha,beta,rho,c_test,k):
        r = np.sqrt(1-c_test**2/alpha**2+0j)
        s = np.sqrt(1-c_test**2/beta**2+0j)
        Cr = np.cosh(k*r*h)
        Sr = np.sinh(k*r*h)
        Cs = np.cosh(k*s*h)
        Ss = np.sinh(k*s*h)
        D = 2*(1-Cr*Cs) + (1/(r*s) + r*s)*Sr*Ss
        k11_e = (k*rho*c_test**2)/D * (s**(-1)*Cr*Ss - r*Sr*Cs)
        k12_e = (k*rho*c_test**2)/D * (Cr*Cs - r*s*Sr*Ss - 1) - k*rho*beta**2*(1+s**2)
        k13_e = (k*rho*c_test**2)/D * (r*Sr - s**(-1)*Ss)
        k14_e = (k*rho*c_test**2)/D * (-Cr + Cs)
        k21_e = k12_e
        k22_e = (k*rho*c_test**2)/D * (r**(-1)*Sr*Cs - s*Cr*Ss)
        k23_e = -k14_e
        k24_e = (k*rho*c_test**2)/D * (-r**(-1)*Sr + s*Ss)
        k31_e = k13_e
        k32_e = k23_e
        k33_e = k11_e
        k34_e = -k12_e
        k41_e = k14_e
        k42_e = k24_e
        k43_e = -k21_e
        k44_e = k22_e
        Ke = np.array([[k11_e, k12_e, k13_e, k14_e],
                        [k21_e, k22_e, k23_e, k24_e],
                        [k31_e, k32_e, k33_e, k34_e],
                        [k41_e, k42_e, k43_e, k44_e]])
        return Ke

    def layer_stiffness_matrix_halfspace(self,alpha,beta,rho,c_test,k):
        r = np.sqrt(1-c_test**2/alpha**2+0j)
        s = np.sqrt(1-c_test**2/beta**2+0j)
        k_11 = k*rho*beta**2*(r*(1-s**2))/(1-r*s)
        k_12 = k*rho*beta**2*(1-s**2)/(1-r*s) - 2*k*rho*beta**2
        k_21 = k_12
        k_22 = k*rho*beta**2*(s*(1-s**2))/(1-r*s)
        Ke_halfspace = np.array([[k_11,k_12],
                                [k_21, k_22]])
        return Ke_halfspace

    def stiffness_matrix(self,c_test,k,h,alpha,beta,rho,n):
        K = np.zeros((2*(n+1),2*(n+1)))
        eps = 0.0001
        while any(abs(c_test-beta) < eps) or any(abs(c_test-alpha) < eps):
            c_test = c_test*(1-eps)
        for j in range(n):
            Ke = self.layer_stiffness_matrix(h[j],alpha[j],beta[j],rho[j],c_test,k)
            DOFs = np.arange(2*j,2*(j+2),1)
            for row in DOFs:
                for col in DOFs:
                    row_idx = np.array(np.where(DOFs==row)).flatten()[0]
                    col_idx = np.array(np.where(DOFs==col)).flatten()[0]
                    K[row,col] = K[row,col] + Ke[row_idx,col_idx]
        Ke_halfspace = self.layer_stiffness_matrix_halfspace(alpha[-1],beta[-1],rho[-1],c_test,k)
        DOFs = np.arange(2*n,2*(n+1),1)
        for row in DOFs:
            for col in DOFs:
                row_idx = np.array(np.where(DOFs==row)).flatten()[0]
                col_idx = np.array(np.where(DOFs==col)).flatten()[0]
                K[row,col] = K[row,col] + Ke_halfspace[row_idx,col_idx]
        Det = np.real(np.linalg.det(K))
        return Det
    def theorectical_dispersion_curve(self):
        c_test = self.c_test
        h = self.h
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        n = self.n
        k = self.wavenumber
        Det = np.zeros((len(k),len(c_test)))
        self.c_t = np.zeros((len(k),1))
        self.lambda_t = np.zeros((len(k),1))
        for l in range(len(k)):
            for m in range(len(c_test)):
                Det[l,m] = self.stiffness_matrix(c_test[m],k[l],h,alpha,beta,rho,n)
                if m==0:
                    sign_old = np.sign(Det[l,m])
                else:
                    sign_old = signD
                signD = np.sign(Det[l,m])
                if sign_old*signD == -1:
                    self.c_t[l] = c_test[m]
                    self.lambda_t[l] = 2*np.pi*(1/k[l])
                    break
        if plot_exp_the_DC == 'Yes':
            fig,ax = plt.subplots(figsize=(4.5,6),dpi=100)
            ax.plot(self.c_exp,self.lambda_exp,'-ob')
            ax.plot(self.c_t,self.lambda_t,'-or')
            ax.set_xlabel('Reyleigh wave velocity [m/s]',fontsize = FigFontSize)
            ax.set_ylabel('Wavelength [m]', fontsize = FigFontSize)
            ax.legend(['Experimentral DC','Theorectical DC'],loc='best')
            plt.gca().invert_yaxis()
            plt.show()
        return self.c_t,self.lambda_t
    def mistfit(self):
        N = len(self.c_t)
        Norm_diff = []
        for i in range(N):
            Norm_diff.append(np.abs(self.c_exp[i]-self.c_t[i])/self.c_exp[i])
        MSE = (1/N)*np.sum(Norm_diff)
        misfitVal = 'Misfit is: {0:.5f}%'.format(100*MSE)
        return misfitVal
        
backward_calculation = seismic_backward(n,beta,h,alpha,rho,c_test,c_curve0,lambda_curve0)
c_t,lambda_t = backward_calculation.theorectical_dispersion_curve()
error = backward_calculation.mistfit()
print(error)