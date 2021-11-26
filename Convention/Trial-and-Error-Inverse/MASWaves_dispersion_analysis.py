import numpy as np
import matplotlib.pyplot as plt
##
FigFontSize = 11
resolution = 100
##
Filename = 'SampleData.txt'
LoadData = np.loadtxt(Filename, dtype='float',delimiter=None)
##
fs = 1000
N=24
x1=10
dx = 1
du=1/75
cT_min = 50
cT_max = 400
delta_cT = 1
fmin = 1
fmax = 50
f_receiver = 4.5
p = 95
up_low_bounds = 'Yes'
Direction = 'Forward'
dispersion_curve = 'lambda_c'
plot_dispersion_curve = 'Yes'
##
class dispersion_analysis(object):
    def __init__(self,fs,N,x1,dx,du,cT_min,cT_max,delta_cT,fmin,fmax,f_receiver):
        self.fs = fs
        self.N = N
        self.x1,self.dx = x1,dx
        self.du = du
        self.cT_min,self.cT_max = cT_min,cT_max
        self.delta_cT = delta_cT
        self.fmin,self.fmax,self.f_receiver = fmin,fmax,f_receiver
        u_read = np.loadtxt(Filename,dtype='float', delimiter=None)
        if Direction == 'Forward':
            self.u = u_read
        elif Direction == 'Backward':
            self.u = np.fliplr(u_read)
        else:
            print('Error')
        self.Tmax = (1/self.fs) * (len(self.u[:,0]) - 1)
        self.T = np.linspace(0,self.Tmax,len(self.u[:,0]))
        self.L = (self.N-1)*self.dx
        self.x = np.arange(self.x1,self.L+x1+1E-05,dx)
        ## plot the waveform
        SignalToPlot = np.zeros((len(self.u[:,0]),self.N),dtype='float')
        fig,ax = plt.subplots(figsize=(5,6),dpi=100)
        for j in range(N):
            SignalToPlot[:,j] = self.u[:,j] + j*self.du*self.dx     
            ax.plot(SignalToPlot[:,j],self.T,'k')
        ax.set_xlim(-5*self.dx*self.du,self.L*self.du+5*self.dx*self.du)
        ax.set_ylim(0,self.Tmax)
        ax.set_xlabel('Distance from source [m]', fontsize = FigFontSize)
        ax.set_ylabel('Time [s]', fontsize = FigFontSize)
        ax.set_xticks(np.arange(0,self.L*self.du+6*dx*self.du,6*dx*self.du))
        plt.gca().invert_yaxis()
        plt.show() 
        ## end plot
    def MASW_Despersion_imaging(self):
        omega_fs = 2*np.pi*self.fs
        Lu = len(self.u[:,0])
        U = np.zeros((Lu,self.N),dtype='complex')
        P = np.zeros((Lu,self.N),dtype='complex')
        Unorm = np.zeros((Lu,self.N),dtype='complex')
        for j in range(self.N):    
            U[:,j] = np.fft.fft(self.u[:,j])
        LU = len(U[:,0])
        for j in range(self.N):
            for k in range(int(LU)):
                Unorm[k,j] = U[k,j] / np.abs(U[k,j])
            P[:,j] = np.exp(1j*(-np.angle(U[:,j])))
        omega = (1/LU) * np.arange(0,LU+1,1)*omega_fs
        cT = np.arange(self.cT_min,self.cT_max+self.delta_cT,self.delta_cT)
        LcT = len(cT)
        self.c = np.zeros((LU, LcT))
        self.f = np.zeros((LU,LcT))
        self.A = np.zeros((LU,LcT))
        for j in range(int(LU)):
            for k in range(int(LcT)):
                self.f[j,k] = omega[j] / (2*np.pi)
                self.c[j,k] = cT[k]
                delta = omega[j]/cT[k]
                temp = 0
                for l in range(self.N):
                    temp = temp + np.exp(-1j*delta*self.x[l])*P[j,l]
                self.A[j,k] = np.abs(temp)/self.N
        
        RemoveMin = np.abs((self.f[:,0] - self.fmin))
        RemoveMax = np.abs((self.f[:,0] - self.fmax))
        IdxMin = np.array(np.where(RemoveMin==np.min(RemoveMin))).flatten()[0]
        IdxMax = np.array(np.where(RemoveMax==np.min(RemoveMax))).flatten()[0]
        self.Aplot = self.A[IdxMin:IdxMax+1,:]
        self.fplot = self.f[IdxMin:IdxMax+1,:]
        self.cplot = self.c[IdxMin:IdxMax+1,:]

        ## plot the dispersion imaging
        fig,ax = plt.subplots(figsize=(4,6),dpi=100)
        plt.contourf(self.cplot,self.fplot,self.Aplot,resolution,cmap=plt.cm.jet)    
        ax.set_xlabel('Phase velocity [m/s]',fontsize = FigFontSize)
        ax.set_ylabel('Frequency (Hz)',fontsize = FigFontSize)
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(orientation='horizontal',location='top')
        cbar.set_label('Normalized amplitude', fontsize = FigFontSize)
        plt.show()
        ## end plot

        self.abs_norm = np.zeros((self.Aplot.shape))
        for i in range(len(self.Aplot[:,0])):
            self.abs_norm[i,:] = np.divide(self.Aplot[i,:],np.max(self.Aplot[i,:]))
        loc = np.argwhere(self.abs_norm.T == 1)
        self.c_loc = loc[:,0]
        self.f_loc = loc[:,1]
        Amax_fvec = np.zeros((len(self.f_loc)))
        Amax_cvec = np.zeros((len(self.c_loc)))
        for i in range(len(self.f_loc)):
            Amax_cvec[i] = self.cplot[0,self.c_loc[i]]
            Amax_fvec[i] = self.fplot[self.f_loc[i],0]   
        ii = np.where(Amax_fvec > self.f_receiver)
        Amax_cvec = Amax_cvec[ii]
        Amax_fvec = Amax_fvec[ii]
        jj = np.argsort(Amax_fvec)
        self.Amax_fvec_sort = Amax_fvec[jj]
        self.Amax_cvec_sort = Amax_cvec[jj]
        ## start input
        lower_freq = int((input('Input lower frequency value: ')))
        higher_freq = int((input('Input higher frequency value: ')))
        ## end input
        nP0 = np.arange(lower_freq,higher_freq+1,1)
        self.f_curve0 = self.Amax_fvec_sort[nP0]
        self.c_curve0 = self.Amax_cvec_sort[nP0]
        self.lambda_curve0 = np.divide(self.c_curve0, self.f_curve0)
        dispersion_data = np.zeros((len(self.c_curve0),2),dtype='object')
        dispersion_data[:,0] = self.c_curve0
        dispersion_data[:,1] = self.lambda_curve0
        np.savetxt('dispersion_data.txt',dispersion_data)   
        
        if up_low_bounds == 'Yes':
            loc_p = np.argwhere(self.abs_norm.T > p/100)
            f_loc_p, c_loc_p = loc_p[:,1], loc_p[:,0]
            Amax_fvec_p = np.zeros((f_loc_p.shape))
            Amax_cvec_p = np.zeros((c_loc_p.shape))
            for i in range(len(f_loc_p)):                
                Amax_fvec_p[i] = self.fplot[f_loc_p[i], 0]
                Amax_cvec_p[i] = self.cplot[0,c_loc_p[i]]
            kk = np.where(Amax_fvec_p > self.f_receiver)
            Amax_fvec_p = Amax_fvec_p[kk]
            Amax_cvec_p = Amax_cvec_p[kk]
            pp = np.argsort(Amax_fvec_p)
            self.Amax_fvec_sort_p = Amax_fvec_p[pp]
            self.Amax_cvec_sort_p = Amax_cvec_p[pp]
            self.Amax_fvec_sort_p_cell = np.empty((len(np.unique(self.Amax_fvec_sort_p)), 1),dtype=object)
            self.Amax_cvec_sort_p_cell = np.empty((len(np.unique(self.Amax_fvec_sort_p)), 1),dtype=object)
            self.f_curve0_up_temp = np.zeros((len(np.unique(self.Amax_fvec_sort_p)),1))
            self.c_curve0_up_temp = np.zeros((len(np.unique(self.Amax_fvec_sort_p)),1))
            self.f_curve0_low_temp = np.zeros((len(np.unique(self.Amax_fvec_sort_p)),1))
            self.c_curve0_low_temp = np.zeros((len(np.unique(self.Amax_fvec_sort_p)),1))      
            U = np.unique(self.Amax_fvec_sort_p, return_index=False) ## return index's sorted order
            for i in range(len(U)):
                self.Amax_fvec_sort_p_cell[i][0] = self.Amax_fvec_sort_p[np.where(self.Amax_fvec_sort_p==U[i])]
                self.Amax_cvec_sort_p_cell[i][0] = self.Amax_cvec_sort_p[np.where(self.Amax_fvec_sort_p==U[i])]           
                self.f_curve0_up_temp[i] = np.max(self.Amax_fvec_sort_p_cell[i][0])
                self.c_curve0_up_temp[i] = np.max(self.Amax_cvec_sort_p_cell[i][0])
                self.f_curve0_low_temp[i] = np.min(self.Amax_fvec_sort_p_cell[i][0])
                self.c_curve0_low_temp[i] = np.min(self.Amax_cvec_sort_p_cell[i][0])
            self.f_curve0_up = self.f_curve0_up_temp[nP0]
            self.c_curve0_up = self.c_curve0_up_temp[nP0]
            self.f_curve0_low = self.f_curve0_low_temp[nP0]
            self.c_curve0_low = self.c_curve0_low_temp[nP0]
            self.lambda_curve0_up = np.divide(self.c_curve0_up, self.f_curve0_up)
            self.lambda_curve0_low = np.divide(self.c_curve0_low, self.f_curve0_low)   
            ## start plot
            fig,ax = plt.subplots(figsize=(4,6),dpi=100)
            plt.contourf(self.cplot,self.fplot,self.Aplot,resolution,cmap=plt.cm.jet) 
            ax.plot(self.Amax_cvec_sort,self.Amax_fvec_sort,'ok',markersize=2) 
            ax.plot(self.c_curve0,self.f_curve0,'ow',markersize=4) 
            # ax.plot(self.Amax_cvec_sort_p,self.Amax_fvec_sort_p,'ok',markersize=2) # Vph at constant freq.
            ax.plot(self.c_curve0_up,self.f_curve0_up, '-ok',markersize=2)
            ax.plot(self.c_curve0_low,self.f_curve0_low,'-ok',markersize=2)
            ax.set_xlabel('Phase velocity [m/s]',fontsize = FigFontSize)
            ax.set_ylabel('Frequency (Hz)',fontsize = FigFontSize)
            plt.gca().invert_yaxis()
            cbar = plt.colorbar(orientation='horizontal',location='top')
            cbar.set_label('Normalized amplitude', fontsize = FigFontSize)
            labels = np.arange(1,len(self.Amax_fvec_sort)+1,1)
            for i, addtext in enumerate(labels):
                ax.annotate(addtext, (self.Amax_fvec_sort[i],self.Amax_cvec_sort[i]),horizontalalignment='right',\
                    verticalalignment='bottom')
            plt.show()
            ## end plot

        ## Plot dispersion curve Rayleigh velocity vs. frequency or wavelegnth
        if dispersion_curve == 'f_c': 
            fig,ax = plt.subplots(figsize=(4.5,6),dpi=100)
            ax.plot(self.f_curve0,self.c_curve0,'o-')
            if up_low_bounds == 'yes':
                ax.plot(self.f_curve0_up,self.c_curve0_up,'r+--')
                ax.plot(self.f_curve0_low,self.c_curve0_low,'r+--')
            ax.set_xlabel('Frequency (Hz)', fontsize = FigFontSize)
            ax.set_ylabel('Reyleigh wave velocity [m/s]', fontsize = FigFontSize)
            ax.legend(['Fundamental DC','Upper BC','Lower BC'],loc='upper right')
        if dispersion_curve == 'lambda_c':
            fig,ax = plt.subplots(figsize=(4.5,6),dpi=100)
            ax.plot(self.c_curve0,self.lambda_curve0, 'o-')
            if up_low_bounds == 'yes':
                ax.plot(self.c_curve0_up,self.lambda_curve0_up,'r+--')
                ax.plot(self.c_curve0_low,self.lambda_curve0_low,'r+--')
            plt.gca().invert_yaxis()
            ax.set_xlabel('Reyleigh wave velocity [m/s]',fontsize = FigFontSize)
            ax.set_ylabel('Wavelength [m]', fontsize = FigFontSize)
            ax.legend(['Fundamental DC','Upper BC','Lower BC'],loc='upper right')
        plt.show()
        return self.c_curve0,self.lambda_curve0
dispersion = dispersion_analysis(fs,N,x1,dx,du,cT_min,cT_max,delta_cT,fmin,fmax,f_receiver)
c_curve0,lambda_curve0 = dispersion.MASW_Despersion_imaging()
