import numpy as np
import sympy as sp

h = 2
rho = 1850
c_test = 10
alpha = 400
beta = 150
k = 1
r = np.sqrt(1-c_test**2/alpha**2)
s = np.sqrt(1-c_test**2/beta**2)
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

Ke = [[k11_e, k12_e, k13_e, k14_e],
      [k21_e, k22_e, k23_e, k24_e],
      [k31_e, k32_e, k33_e, k34_e],
      [k41_e, k42_e, k43_e, k44_e]]
# print(Ke)
for j in range(4):
    print(np.arange(2*j,2*j+3+1,1))