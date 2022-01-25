import numpy as np
import sympy as sp
import tkinter
import matplotlib.pyplot as plt
from pandas import *

shearWave = [150,300,400]
thickness = [5,10,np.inf]

lambda_min = 2
lambda_max = 40
delta_lambda = 4

wl = np.arange(lambda_min,lambda_max+delta_lambda,delta_lambda)
