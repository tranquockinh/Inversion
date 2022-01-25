
from backward import *

bw = backward(wl,shearWave,thickness,1/3,len(wl))
#%% ploting data

_,invthickness = bw.initialization()
inverted_Vs = bw.inversion()
checkSwave,checkRwave = bw.check_forward()
test_Rwave = bw.parameters()
iniSwave = shearWave
iniThickness = thickness
invSwave = inverted_Vs

def plot(iniSwave,iniThickness,invSwave,invThickness,test_Rwave,checkRwave,wl):
    
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

    ax[1].plot(test_Rwave,wl,'-or',linewidth=2,markerfacecolor='r',label='Inverted')
    ax[1].plot(checkRwave,wl,'-ob',linewidth=2,markerfacecolor='None',label='Check')
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Phase velocity, Vph [m/s]')
    ax[1].set_ylabel('Wavelength, [m]')
    ax[1].xaxis.set_label_position('top')
    ax[1].xaxis.tick_top()
    ax[1].spines['bottom'].set_color('white')
    ax[1].spines['right'].set_color('white')
    ax[1].legend()
    plt.show()

plot(iniSwave,iniThickness,invSwave,invthickness,test_Rwave,checkRwave,wl)
