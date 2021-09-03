import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# DISPERSION
Layer_Data = np.array([[100,5],
                       [200,np.inf]])
# Arrange vectors of shear wave velocity and depth
Depth = np.append(0,Layer_Data[:,1])
num_layer = len(Depth) - 1
Vs = Layer_Data[:,0]
# Poisson's ratio
PR = 0.3
# Initialize wavelength
DC_points = 15
Lambda = np.zeros((DC_points))
Delta_Lambda = 2
for i in range(DC_points):
    if i == 0:
        Lambda[0] = 2
    else:
        Lambda[i] = Lambda[i-1] + Delta_Lambda
# Function to compute weights
def weighting(D,WL):
    z = sp.symbols('z')
    material_coefficient = 1
    limit_low = D[:-1]
    limit_up = D[1:]
    cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi, -0.3933*2*np.pi])
    cv1 = cv[0] 
    cv2 = cv[1] 
    cv3 = cv[2] 
    cv4 = cv[3]
    # Particle displacement function
    PDF = (cv1*sp.exp(cv3/WL*z) + cv2*sp.exp(cv4/WL*z))*material_coefficient
    # Loop to compute wave a the wavelength given
    Area_i = np.zeros((num_layer),dtype='object')
    # Total area
    Area = sp.integrate(PDF,(z,0,np.inf))
    for j in range(num_layer):
        Area_i[j] = sp.integrate(PDF,(z,limit_low[j],limit_up[j]))
    weights = Area_i / Area
    return weights
# Computing phase velocity by weighted average at each wavelength
beta = (0.87+1.12*PR)/(1+PR)
Vph = np.zeros((DC_points))
for i in range(DC_points):
    wp = weighting(Depth,Lambda[i])
    Vph[i] = np.dot(beta*Vs,wp[i].flatten())
    print(('WAVELENGTH{}---WEIGHTING = {}---PHASE VELOCITY = {}').format(i+1,wp,Vph[i]))
fig,ax = plt.subplots(1,2,figsize=(6,6),dpi=100)
ax[0].plot(Vph,Lambda,'-o',markerfacecolor='None')
ax[0].invert_yaxis()
ax[0].set_xlabel('Phase velocity, Vph [m/s]')
ax[0].set_ylabel('Wavelength, [m]')
ax[0].xaxis.set_label_position('top')
ax[0].xaxis.tick_top()
ax[0].set_xlim(0,Vs[-1])
ax[0].spines['bottom'].set_color('white')
ax[0].spines['right'].set_color('white')
# INVERSION
# Srating from data from dispersion curve
# Data include: Phase velocity, Wavelength, Alpha
Alpha = 0.2
Step_Z = 0
Step_Vph = []
Vs = np.zeros((DC_points))
# Suppose I have only one layer
# Adding layers
for i in range(DC_points):
    Step_Z = np.append(Step_Z,Alpha*Lambda[i])
    Depth = np.append(Step_Z,np.inf)
    weighting_factor,num_layer = weighting(Depth,Lambda[i])
    Step_Vph = np.append(Vph[:i+1],Vph[i])
    Vs[i] = np.dot((1/beta)*Step_Vph,weighting_factor)
    print(('LAYER{}---PHASE VELOCITY = {}---SHEAR WAVE VELOCITY = {}').format(i+1,Vph[i],Vs[i]))
    ax[1].step(Vs[:i],Step_Z[:i],'r',markerfacecolor='None')
    ax[1].invert_yaxis()
    ax[1].set_xlabel('shear wave velocity, Vs [m/s]')
    ax[1].set_ylabel('Wavelength, [m]')
    ax[1].xaxis.set_label_position('top')
    ax[1].xaxis.tick_top()
    ax[1].set_xlim(0,Vs[-1])
    ax[1].spines['bottom'].set_color('white')
    ax[1].spines['right'].set_color('white')
plt.show()