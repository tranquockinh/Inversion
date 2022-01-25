#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Parameters

# the input data is the dispersion properties (phase velocity is a function of wavelegnth) 
# of the ground layers
# let consider a layered system with several layers, the theorectical dispersion
# curve can be computed using transfer matrix method, or other method such as original
# Thomson-Hankel method

# The we could be able to compute the waveform using the dispersion data

df = pd.read_excel("DispData_INPUT.xls",sheet_name="List1",
                    dtype={'Name': str, 'Value': float})
data = df.to_numpy()
max_x = 101
spacing = 2
delta_impulse = 300 # F(t) in ms
T = 1000
sampling_interval = 1 
Z = 2
damping = 1/8000
freq = data[:,0]
Vph = data[:,2]
num_freq = len(freq)
omega = 2*np.pi*freq/1000
x = np.arange(1,max_x+spacing,spacing)
num_cycles = max_x / spacing

# %%
t = np.arange(0,delta_impulse+sampling_interval,sampling_interval)
damping_var = np.zeros((num_freq))
F = np.zeros((num_freq,len(t)))
F_var = np.zeros((num_freq,len(t)))

for i in range(num_freq):
    damping_var[i] = 1*freq[i]*damping
    for j in range(len(t)):
        F[i,j] = np.exp(-1*damping*t[j]) * np.sin(omega[i]*t[j])
        F_var[i,j] = np.exp(-1*damping_var[i]*t[j]) * np.sin(omega[i]*t[j])

fig,ax = plt.subplots(figsize=(8,5))
for i in range(num_freq):
    max_trace = np.max(F[i,:])
    min_trace = np.min(F[i,:])
    # max_amplitude = np.max(max_trace,np.abs(min_trace))
    ax.plot(t,F_var[i,:]*2 + freq[i])
ax.set_title('Varying damping ratio')
ax.set_xlabel('Time (ms)',fontsize=12)
ax.set_ylabel('Damped frequency (Hz)',fontsize=12)
plt.show()
#%% Synthetic waveform
num_waveform = int(max_x/spacing)
t_start = np.zeros((num_freq,num_waveform))
t_start2 = np.zeros((num_freq,num_waveform))
z_start = np.zeros((num_freq))
z_end = np.zeros((num_freq))
fig,ax = plt.subplots(figsize=(5.5,8))
for j in range(num_waveform):
    t_start[:num_freq,j] = x[j] / Vph
    t_total = np.arange(0,T+sampling_interval,sampling_interval)
    vel_z = T*(1/sampling_interval) + 1
    sum_F = np.zeros((int(vel_z),num_waveform))
    z = np.zeros((num_freq,int(vel_z)))
    t_start2[:num_freq,j] = np.round(t_start[:,j]*(1/sampling_interval))
    
    for i in range(num_freq):
        z_start[i] = t_start2[i,j]
        z_end[i] = t_start2[i,j]+delta_impulse*(1/sampling_interval)
        z[i,int(z_start[i]):int(z_end[i])+1] = F_var[i,:]
    sum_F[:,j] = sum(z)
    max_amp = np.max(sum_F[:,j])
    min_amp = np.min(sum_F[:,j])
    max_plot = max(max_amp,np.abs(min_amp))
    ax.plot(sum_F[:,j]*Z/max_plot+j*spacing,t_total,'-k',linewidth=2)
ax.invert_yaxis()
ax.set_xlabel('Offset',fontsize=12)
ax.set_ylabel('Time (ms)',fontsize=12)
ax.set_title('Synthetic waveform, f(t,x)')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
# %% Plot spectrum
t,x = np.shape(sum_F)
max_f = 1/(2*sampling_interval/1000) # from kHz rto Hz
f_increment = 1/(t_total[-1]/1000) # from ms to s
f_array = np.arange(0,max_f+f_increment,f_increment)
mean_sum_F = np.mean(sum_F,1)
mean_mean_sum_F = np.mean(mean_sum_F)
rm_sum_F = sum_F - mean_mean_sum_F
spectrum = np.abs(np.fft.fft2(rm_sum_F))
half_spectrum = spectrum[:int((t+1)/2),:]
normalized_spectrum = np.zeros((len(half_spectrum),x))
for i in range(x):
    normalized_spectrum[:,i] = half_spectrum[:,i]/np.max(spectrum[:,i])
# Range of frequency spectrum to be draw
fig,ax = plt.subplots(figsize=(8,5))
fstart = 2
f_end = 180
Gain = 10
for i in range(x):
    ax.plot(f_array,normalized_spectrum[:,i]*Gain+i)
ax.set_xlabel('Frequency (hz)',fontsize=12)
ax.set_ylabel('Offset (m)',fontsize=12)
ax.set_title('Power spectrum')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()