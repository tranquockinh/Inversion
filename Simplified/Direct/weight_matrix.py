from data import *


def particle_disp_1(up, low, wavelength):

    z = sp.Symbol('z')
    c1 = 0.13
    alpha = -2 * np.pi / wavelength
    c2 = -0.24
    beta = -0.3 * (2*np.pi) / wavelength
    
    displ_func = c1/alpha * sp.exp(alpha * z) + c2/beta * sp.exp(beta * z)

    displ_up = displ_func.subs(z, up)
    displ_low = displ_func.subs(z, low)
    displ = displ_up - displ_low

    return displ


def particle_disp_2(up, low, wavelength):

    z = sp.Symbol('z')

    displ_func = z - (2*wavelength/5) * (z/wavelength)**(5/2)

    displ_up = displ_func.subs(z, up)
    displ_low = displ_func.subs(z, low)
    displ = displ_low - displ_up

    return displ


def input_profile(depth):

    depth = np.array(depth)
    
    if (depth[0] != 0):
        depth = np.append(0,depth)
    if (depth[-1] != np.inf):
        depth[-1] = 1E6

    return depth


def vector_displacement(which_function, depth, wavelength):



    depth = input_profile(depth)
    upper = depth[:-1]
    lower = depth[1:]

    num_layer = len(depth) - 1

    displ = np.zeros((num_layer))
    norm_vector_displ = np.zeros((num_layer))
    
    i = 0

    if (which_function == 1):

        for up, low in zip(upper, lower):

            if (low <= wavelength): 
                int_up = up
                int_low = low
                displ[i] = particle_disp_2(int_up, int_low, wavelength)
     
            elif (low > wavelength):
                int_up = up
                int_low = wavelength
                displ[i] = particle_disp_2(int_up, int_low, wavelength)

                break

            norm_vector_displ[i] = displ[i] / displ[0]
            i += 1

    elif (which_function == 2):

        for up, low in zip(upper, lower):

            displ[i] = particle_disp_1(up, low, wavelength)
            norm_vector_displ[i] = displ[i] / displ[0]

            i += 1

    return displ, norm_vector_displ



def weight_matrix(which_function, depth, list_wavelength):
    
    
    weight_mat = np.zeros((len(list_wavelength), len(depth)))
    
    for j, wavelength in enumerate(list_wavelength):

        disp, n_disp = vector_displacement(which_function, depth, wavelength)
        weight = [disp[i]/sum(disp) for i in range(len(disp))]
        weight_mat[j,:] = weight

    return weight_mat




list_wavelength = np.arange(2,10,0.68)

def forward(which_function, coeff, depth, shear_wave, list_wavelength):

    weight_mat = weight_matrix(which_function, depth, list_wavelength)
    Rayleigh_wave = coeff * np.matmul(weight_mat, shear_wave)

    return Rayleigh_wave

Rayleigh_wave = forward(2, 0.93, depth, shear_wave, list_wavelength)

def conversion_parameter(list_wavelength, Rayleigh_wave, alpha, coeff):

    cdepth = alpha * list_wavelength
    cshear_wave = coeff * Rayleigh_wave

    return cdepth, cshear_wave


def iter_inversion(which_function, list_wavelength, Rayleigh_wave, alpha, coeff, num_ilayer):

    if (num_ilayer <= len(list_wavelength)):
    
        cdepth, cshear_wave = conversion_parameter(list_wavelength, Rayleigh_wave, alpha, coeff)
        idepth = cdepth [:num_ilayer]
        iwavelength = list_wavelength[:num_ilayer]
        ishear_wave_data = cshear_wave[:num_ilayer]


        weight_mat = weight_matrix(which_function, idepth, iwavelength)

    else: 
        weight_mat = 'Error: Number of inverted layers is more than the length of wavelength'


    U,S,VT = np.linalg.svd(weight_mat, full_matrices = False)
    iweight_mat = np.matmul(np.matmul(VT.T,np.linalg.inv(np.diag(S))),U.T)

    ishear_wave = np.matmul(iweight_mat, ishear_wave_data)

    return ishear_wave

ishear_wave = iter_inversion(2, list_wavelength, Rayleigh_wave, 1.5, 0.93, 8)

print(ishear_wave)
