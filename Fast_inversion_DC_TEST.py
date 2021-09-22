import numpy as np
import sympy as sp
import tkinter
import matplotlib
import matplotlib.pyplot as plt
# DISPERSION
Layer_Data = np.array([[100,2],                   
                       [150,5],
                       [200,10],
                       [300,20],
                       [400,np.inf]])
# Arrange vectors of shear wave velocity and depth
numLayers = len(Layer_Data[:,0])
Depth = np.append(0,Layer_Data[:,1])
Vs = Layer_Data[:,0]
# Poisson's ratio
PR = 0.3
# Initialize wavelength

'''
Lambda = np.zeros((DC_points))
Delta_Lambda = 5
for i in range(DC_points):
    if i == 0:
        Lambda[0] = 2
    else:
        Lambda[i] = Lambda[i-1] + Delta_Lambda
'''
#Lambda = np.array([2,5,10,100])
Lambda = np.array([2,3,4,5,6,7,8,9,10,11,13,15,17,20,100])
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
        Area_i[j] = sp.integrate(PDF,(z,limit_low[j],limit_up[j]))
    weights = Area_i / Area
    return weights
# Computing phase velocity by weighted average at each wavelength
beta = (0.87+1.12*PR)/(1+PR)
Vph = np.zeros((DC_points))
WEIGHTS = np.zeros((numLayers,DC_points))
for i in range(DC_points):
    weights = weighting(Depth,Lambda[i])
    Vph[i] = np.dot(beta*Vs,weights.flatten())
    print(('({})-WEIGHTING = {}\nPHASE VELOCITY = {}').format(i+1,weights,Vph[i]))
    WEIGHTS[:,i] = weights
print('weights\n',WEIGHTS)

def SVD(WEIGHT_MATRIX):
    U,s,VT = np.linalg.svd(WEIGHT_MATRIX,full_matrices=False)
    V = VT.T
    UT = U.T
    inv_s = np.linalg.inv(np.diag(s))
    inv_A_SVD = np.dot(V,np.dot(inv_s,UT))
    return U,s,VT,inv_A_SVD
'''
WEIGHT_MATRIX = np.zeros((numLayers,DC_points))
for i in range(DC_points):
    weights = weighting(Depth,Lambda[i])
    WEIGHT_MATRIX[:,i] = weights
    print(('LWAVELENGTH {} PASSED').format(i+1))
print(WEIGHT_MATRIX)
U,s,VT,inv_A_SVD = SVD(WEIGHT_MATRIX)
RE_CONSTRUCTION = np.matmul(inv_A_SVD.T,Vph*(1/beta))
print()
print(RE_CONSTRUCTION)
'''
numLayers_refine = DC_points
Bottoms = []
for i in range(numLayers_refine):
    Coeff = 1 # NEARLY 1/3 OF WAVELENGTH
    Current_bottom = Coeff*Lambda[i]
    Bottoms = np.append(Bottoms,Current_bottom)
Bottoms[-1] = np.inf
Depths = np.append(0,Bottoms)

# DO NOT CHANGE # LAYERS AND THICKNESSES 
# COMPUTE WEIGHTS AT EACH WAVELENGTH
WEIGHT_MATRIX = np.zeros((numLayers_refine,DC_points))
for i in range(DC_points):
    weights = weighting(Depths,Lambda[i])
    WEIGHT_MATRIX[:,i] = weights
    print(('WAVELENGTH {} PASSED').format(i+1))
print(WEIGHT_MATRIX)

U,s,VT,inv_A_SVD = SVD(WEIGHT_MATRIX)
RE_CONSTRUCTION = np.matmul(inv_A_SVD.T,Vph*(1/beta))
print()
print(RE_CONSTRUCTION)