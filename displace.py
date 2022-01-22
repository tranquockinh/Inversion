
from data import *

depth = np.append(0,depth)
depth = np.arange(0,10,1)

def boundary(depth):
    
    z = sp.Symbol('z')
    upper = depth[:-1]
    lower = depth[1:]

    return z, upper, lower


def particle_disp(wavelength):
    
    z, upper, lower = boundary(depth)
    c1 = 0.13
    alpha = -2 * np.pi / wavelength
    c2 = -0.24
    beta = -0.3 * (2*np.pi) / wavelength
    
    displ_func = c1/alpha * sp.exp(alpha * z) + c2/beta * sp.exp(beta * z)
    total_disp_up = displ_func.subs(z, 0)
    total_disp_low = displ_func.subs(z, np.inf)
    total_displ = total_disp_up - total_disp_low

    portion_displ = []
    weight = []
    i = 0
    for up, low in zip(upper, lower):
        portion_displ_up = displ_func.subs(z, up)
        portion_displ_low = displ_func.subs(z, low)
        portion_displ.append(portion_displ_up - portion_displ_low)
        weight.append(portion_displ[i] / total_displ)
        i += 1
    # print(sum(weight))

    return weight

weight = particle_disp(10)

def particle_disp2(wavelength):

    z, upper, lower = boundary(depth)
    displ_func = z - (2*wavelength/5) * (z/wavelength)**(5/2)
    total_disp_up = displ_func.subs(z, 0)
    total_disp_low = displ_func.subs(z, wavelength)
    total_displ = total_disp_low - total_disp_up

    portion_displ = []
    weight = []
    i = 0
    for up, low in zip(upper, lower):

        if (low <= wavelength):
            portion_displ_up = displ_func.subs(z, up)
            portion_displ_low = displ_func.subs(z, low)
            portion_displ.append(portion_displ_low - portion_displ_up)
            weight.append(portion_displ[i] / total_displ)
            i += 1

        else:
            portion_displ_up = displ_func.subs(z, up)
            tail = wavelength - up            
            portion_displ_low = displ_func.subs(z, up + tail)
            portion_displ.append(portion_displ_low - portion_displ_up)
            weight.append(portion_displ[i] / total_displ)
            i += 1
            break
        
    while i < (len(depth) - 1):
        portion_displ.append(0)
        weight.append(portion_displ[i] / total_displ)
        i += 1 
    
    return weight

weight2 = particle_disp2(10)
fig, ax = plt.subplots(figsize=(4,5))
ax.plot(weight,depth[1:])
ax.plot(weight2,depth[1:])
ax.invert_yaxis()
plt.show()

print(weight)