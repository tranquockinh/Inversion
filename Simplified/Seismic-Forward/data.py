import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *
# shear wave velocity
Swave = [150,200,400]
# thickness
thickness = [6,10,np.inf]

depth = np.append(0,thickness)
nLayer = len(thickness)
# mininum, maximum and stepping wavelength
wl_min = 2
wl_max = 30
wl_step = 4
wavelen = np.arange(wl_min,wl_max+wl_step,wl_step)
# Poisson's ratio
nuy = 0.3
# shear- to compression-wave coefficinet
spr = 0.93
# depth to wavelength ratio
dwr = 1/2
# number of layer inverted
n_inverted_layer = len(wavelen)