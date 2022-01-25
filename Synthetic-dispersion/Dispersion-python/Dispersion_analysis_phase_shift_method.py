#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

data = sio.loadmat('SampleData.mat')
sum_F = [[element for element in upperElement] 
                                for upperElement in data['suma_sg']]
t_total =  [[element for element in upperElement] 
                                for upperElement in data['t_total']]
sum_F = np.array(sum_F)
t_total = np.array(t_total)

# %% Parameters
U = sum_F
print(U.shape)
ct = np.arange(300,1200+1,1)
freq_limit = [15, 80]

fs = 1000 # sampling frequency (T in the synthetic waveform file)
max_x = 101
dx = 2 # spacing in synthetic waveform file
x = np.arange(1,max_x,dx)
time_length,num_trace = np.shape(U)
t = np.arange(1/fs,time_length/fs+1/fs,1/fs)
print(len(t))
print(num_trace,time_length)
sig_plot = np.zeros((time_length,num_trace))
fig,ax = plt.subplots()
for j in range(num_trace):
    sig_plot[:,j] = U[:,j]/50 + j*dx
    ax.plot(sig_plot[:,j],t,'-k',linewidth=2)
ax.invert_yaxis()
ax.set_xlabel('Offset',fontsize=12)
ax.set_ylabel('Time (ms)',fontsize=12)
ax.set_title('Synthetic waveform, f(t,x)')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#%% plot waveform contour in time domain
fig,ax = plt.subplots()
plt.contourf(x,t,U,resolution=100,cmap=plt.cm.jet)
ax.invert_yaxis()
ax.set_xlabel('Offset',fontsize=12)
ax.set_ylabel('Time (ms)',fontsize=12)
ax.set_title('Synthetic waveform, f(t,x)')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
#%% Dispersion image analysis
f = np.linspace(0,len(t),fs)
f = f[:int(np.floor(len(f)/2))]
f_components = len(f)
Ufft = np.fft.fft(U,axis=0)
Ufft = Ufft[:int(np.floor(len(Ufft[:,0])/2)),:]
omega = 2*np.pi*f
Rnorm = Ufft/np.abs(Ufft) # Spectrum normalization 
dispersion_img = np.zeros((f_components,len(ct)))
sum_As = np.zeros((f_components,len(ct)))
for i in range(f_components):
    freq,junk = np.meshgrid(Rnorm[i,:],ct) # A matrix of a single normalized spectrum term
    X,Y = np.meshgrid(x,ct) # A matrix of positions and velocites
    As = np.exp(1j*omega[i]*X/Y)*freq # As(ct)
    sum_As[i,:] = np.abs(sum(np.rot90(As)))
# Solve for phase velocity using the slope of phase angle and position
angUfft = np.angle(Ufft)
unwraping = np.abs(np.unwrap(angUfft.T))
phase_velo_init = np.zeros((f_components))
for j in range(f_components):
    pfit = np.polyfit(x.T,unwraping[:,j],1)
    phase_velo_init[j] = pfit[0] # slope of phase change
phase_velocity = 1/(phase_velo_init/(2*np.pi*f))
fig,ax = plt.subplots()
plt.contourf(ct,f,sum_As,resolution=100,cmap=plt.cm.jet)

#%% dispersion analysis
fig,ax = plt.subplots(figsize=(4,6))
plt.contourf(ct,f,sum_As,resolution=100,cmap=plt.cm.jet)
# cbar = plt.colorbar(orientation='vertical',location='right')
# cbar.set_label('Normalized amplitude', fontsize = 12)
ax.plot(phase_velocity[::2],f[::2],'o-k',linewidth=2)
plt.gca().set_ylim(freq_limit)
ax.set_xlabel('Phase velocity (m/s)',fontsize=12)
ax.set_ylabel('Frequency (hz)',fontsize=12)
ax.set_title('Dispersion curve')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.show()
# %%
