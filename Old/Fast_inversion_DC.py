import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# DISPERSION

Layer_Data = np.array([[100,2],                   
                       [150,5],
                       [200,10],
                       [400,np.inf]])

#Layer_Data = np.array([[100,np.inf]])
# Arrange vectors of shear wave velocity and depth
numLayers = len(Layer_Data[:,0])
Depth = np.append(0,Layer_Data[:,1])
Vs = Layer_Data[:,0]
# Poisson's ratio
PR = 0.3
# Initialize wavelength
'''
DC_points = 10
Lambda = np.zeros((DC_points))
Delta_Lambda = 4
for i in range(DC_points):
    if i == 0:
        Lambda[0] = 4
    else:
        Lambda[i] = Lambda[i-1] + Delta_Lambda
'''
Lambda = np.array([2,3,4,5,6,7,8,9,10,11,13,15,17,20,22])
DC_points = len(Lambda)
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
    num_layer = len(D) - 1
    Area_i = np.zeros((num_layer),dtype='object')
    # Total area
    Area = sp.integrate(PDF,(z,0,np.inf))
    for j in range(num_layer):
        # Area_i[j] = sp.integrate(PDF,(z,limit_low[j],limit_up[j]))
        term1 = (cv1/(cv3/WL))*(sp.exp(cv3/WL*limit_up[j]) - sp.exp(cv3/WL*limit_low[j]))
        term2 = (cv2/(cv4/WL))*(sp.exp(cv4/WL*limit_up[j]) - sp.exp(cv4/WL*limit_low[j]))
        Area_i[j] = term1 + term2
    weights = Area_i / Area
    return weights
# Computing phase velocity by weighted average at each wavelength
beta = (0.87+1.12*PR)/(1+PR)
Vph = np.zeros((DC_points))
WEIGHTS = np.zeros((DC_points,numLayers))
XLabels = np.zeros((DC_points),dtype='object')
YLabels = np.zeros((numLayers),dtype='object')
for i in range(DC_points):
    weights = weighting(Depth,Lambda[i])
    Vph[i] = np.dot(beta*Vs,weights.flatten())
    print(('({})-WEIGHTING = {}\nPHASE VELOCITY = {}').format(i+1,weights,Vph[i]))
    WEIGHTS[i,:] = weights
print('weights\n',WEIGHTS)
# PLOT WEIGHTING MAP AND RE-CONSTRUCTION OF LAYER PROFILE
'''
fig,ax = plt.subplots()
ax.matshow(WEIGHTS, cmap=plt.cm.Blues)
for i in range(DC_points):
    for j in range(numLayers):
        text = ('{0:.3f}').format(WEIGHTS[i,j])
        ax.text(j, i, str(text), va='center', ha='center')
plt.show()
'''
def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    inv_A_SVD = np.dot(V,np.dot(inv_s,UT))
    return U,s,VT,inv_A_SVD
#U,s,VT,inv_A_SVD = SVD(WEIGHTS)
#RE_CONSTRUCTION = np.matmul(inv_A_SVD.T,Vph*(1/beta))
fig,ax = plt.subplots(figsize=(3,5),dpi=100)
ax.plot(Vph,Lambda,'-bo',markerfacecolor='None')
ax.invert_yaxis()
ax.set_xlabel('Phase velocity, Vph [m/s]')
ax.set_ylabel('Wavelength, [m]')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlim(0,Vs[-1])
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
plt.show()
print('-'*10,'COMPLETE DISPERSION ANALYSIS -->INVERSION','-'*10)
'''
# CHECK THE MODEL IF IT WORKS WITH SVD
WEIGHT_MATRIX = np.zeros((numLayers,DC_points))
for i in range(DC_points):
    weights = weighting(Depth,Lambda[i])
    WEIGHT_MATRIX[:,i] = weights
    print(('LWAVELENGTH {} PASSED').format(i+1))
print(WEIGHT_MATRIX)
U,s,VT,inv_A_SVD = SVD(WEIGHT_MATRIX)
RE_CONSTRUCTION = np.matmul(inv_A_SVD.T,Vph*(1/beta))
print(inv_A_SVD)
print()
print(RE_CONSTRUCTION)
'''
# DO INVERSION BASED ON DISPERSION CURVE
numLayers = DC_points
Bottoms = []
for i in range(numLayers):
    Coeff = 1 # NEARLY 1/3 OF WAVELENGTH
    Current_bottom = Coeff*Lambda[i]
    Bottoms = np.append(Bottoms,Current_bottom)
Bottoms[-1] = np.inf
Depths = np.append(0,Bottoms)

# DO NOT CHANGE # LAYERS AND THICKNESSES 
# COMPUTE WEIGHTS AT EACH WAVELENGTH
WEIGHT_MATRIX = np.zeros((DC_points,numLayers))
for i in range(DC_points):
    weights = weighting(Depths,Lambda[i])
    WEIGHT_MATRIX[i,:] = weights
    print(('LWAVELENGTH {} PASSED').format(i+1))
print(WEIGHT_MATRIX)

U,s,VT,inv_A_SVD = SVD(WEIGHT_MATRIX)
RE_CONSTRUCTION = np.matmul(inv_A_SVD,Vph*(1/beta))
print(inv_A_SVD)
print()
print(RE_CONSTRUCTION)
fig,ax = plt.subplots(figsize=(3,5),dpi=100)
ax.step(RE_CONSTRUCTION,Lambda,'-bo',markerfacecolor='None')
ax.invert_yaxis()
ax.set_xlabel('Phase velocity, Vph [m/s]')
ax.set_ylabel('Wavelength, [m]')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
plt.show()
