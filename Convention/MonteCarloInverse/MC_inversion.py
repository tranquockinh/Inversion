import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##
FigFontSize = 11
resolution = 100
##
Filename = 'SampleData_MASWavesInversion.csv'
# sample_data = pd.read_csv(Filename,encoding="utf-8")
sample_data = np.genfromtxt(Filename,delimiter=',',dtype=None)
c_OBS = sample_data[:,0]
up_low_boundary = 'Yes'
c_OBS_low = sample_data[:,1]
c_OBS_up = sample_data[:,2]
lambda_OBS = sample_data[:,3]
c_min, c_max = 75, 300
c_step = 0.09
delta_c = 2
n = 4
h_initial = [2,2,4,6]
beta_initial = [1.09*c_OBS[0], 1.09*c_OBS[9], 1.09*c_OBS[16], 1.09*c_OBS[22], 1.09*c_OBS[-1]]
n_unsat = 1
nu_unsat = 0.3
alpha_temp = [np.sqrt(2*(1-nu_unsat)/(1-2*nu_unsat)) * beta_initial[i] for i in range(len(beta_initial))]
alpha_initial = [alpha_temp[0], 1440, 1440, 1440, 1440]
rho = [1850, 1900, 1950, 1950, 1950]

N_reversals = 0 # Normally dispersive analysis
# Search control parameters
b_S = 5
b_h = 10
N_max = 100
e_max = 1/100

def Fast_delta_recersion(c,k,alpha,beta1,beta2,rho1,rho2,h,X):
    gamma1 = beta1**2/c**2
    gamma2 = beta2**2/c**2
    epsilon = rho2/rho1
    eta = 2*(gamma1-epsilon*gamma2)
    a = epsilon+eta
    ak = a-1
    b = 1-eta
    bk = b-1
    # Layer eigenfunctions
    r = np.sqrt(1-(c**2/alpha**2)+0j)
    C_alpha = np.cosh(k*r*h)
    S_alpha = np.sinh(k*r*h)

    s = np.sqrt(1-(c**2/beta1**2)+0j)
    C_beta = np.cosh(k*s*h)
    S_beta = np.sinh(k*s*h)
     # Layer recursion
    p1 = C_beta*X[1]+s*S_beta*X[2]
    p2 = C_beta*X[3]+s*S_beta*X[4]
    p3 = (1/s)*S_beta*X[1]+C_beta*X[2]
    p4 = (1/s)*S_beta*X[3]+C_beta*X[4]

    q1 = C_alpha*p1-r*S_alpha*p2
    q2 = -(1/r)*S_alpha*p3+C_alpha*p4
    q3 = C_alpha*p3-r*S_alpha*p4
    q4 = -(1/r)*S_alpha*p1+C_alpha*p2

    y1 = ak*X[0]+a*q1
    y2 = a*X[0]+ak*q2

    z1 = b*X[0]+bk*q1
    z2 = bk*X[0]+b*q2

    X[0] = bk*y1+b*y2
    X[1] = a*y1+ak*y2
    X[2] = epsilon*q3
    X[3] = epsilon*q4
    X[4] = bk*z1+b*z2

    return X

def Fast_delta_matrix(c,k,n,alpha,beta,rho,h):
    # initialization
    X_temp = np.array([2*(2-c**2/beta[0]**2), -(2-c**2/beta[0]**2)**2, 0, 0, -4])
    X = [(rho[0]*beta[0]**2)**2 * X_temp[i] for i in range(len(X_temp))]
    # layer recursion
    for i in range(n):
        X = Fast_delta_recersion(c,k,alpha[i],beta[i],beta[i+1],rho[i],rho[i+1],h[i],X)
    # dispersion function value
    r = np.sqrt(1-c**2/alpha[n]**2 + 0j)
    s = np.sqrt(1-c**2/beta[n]**2 + 0j)
    F = (X[1] + s*X[2]) - r * (X[3] + s*X[4])
    return F

def Theorectical_dispersion_curve(c_min,c_max,c_step,Lambda,n,alpha,beta,rho,h,delta_c):
    
    c_test = np.arange(c_min,c_max+c_step,c_step)
    k = (2*np.pi) / Lambda
    modes = 1
    D = np.zeros((len(k),len(c_test)))
    c_t = np.zeros((len(k),modes))
    lambda_t = np.zeros((len(k),modes))
    m_loc = 1
    delta_m = round(delta_c/c_step)

    for j in range(len(k)):
            for m in range(m_loc-1,len(c_test)):
                D[j,m] = Fast_delta_matrix(c_test[m],k[j],n,alpha,beta,rho,h)
                if m == (m_loc-1):
                    sign_old = np.sign(Fast_delta_matrix(c_test[0],k[j],n,alpha,beta,rho,h))
                else:
                    sign_old = signD
                signD = np.sign(D[j,m])
                if sign_old * signD == -1:
                    c_t[j] = c_test[m]
                    lambda_t[j] = 2*np.pi/k[j]
                    m_loc = m - delta_m
                    if m_loc <= 0:
                        m_loc = 1
                    break
    return c_t,lambda_t

def Misfit(c_t,c_OBS):
        Q = len(c_t)
        e = (1/Q) * sum([np.abs(i-j)/i for i,j in zip(c_OBS,c_t)]) 
        return e

def Monte_Carlo_Analysis(c_min,c_max,c_step,delta_c,n,n_unsat,alpha_initial,nu_unsat,beta_initial,
                rho,h_initial,N_reversals,c_OBS,lambda_OBS,up_low_boundary,c_OBS_up,c_OBS_low,
                b_S,b_h,N_max,e_max):
    
    c_t, lambda_t = Theorectical_dispersion_curve(c_min,c_max,c_step,lambda_OBS,n,alpha_initial,
                beta_initial,rho,h_initial,delta_c)
    e_initial = Misfit(c_t,c_OBS)
    store_all = np.zeros((6,N_max),dtype = 'object')
    beta_opt = beta_initial
    h_opt = h_initial
    e_opt = e_initial
    Nend = N_max
    x_data = []
    y_data = []
    for w in range(N_max):
        
        low_rand = [-(b_S/100)*beta_opt[i] for i in range(len(beta_initial))]
        up_rand = [b_S/100*beta_opt[i] for i in range(len(beta_initial))]
        
        beta_test = beta_opt + np.random.uniform(low_rand,up_rand)

        # print((beta_test[N_reversals+1:-1] <= beta_test[N_reversals+2:]).all())
        while ((beta_test[N_reversals+1:-1] <= beta_test[N_reversals+2:]).all()) == False:
            beta_test = beta_test = beta_opt + np.random.uniform(low_rand,up_rand)
        
        alpha_unsat = [np.sqrt((2*(1-nu_unsat)) / (1-2*nu_unsat)) * beta_test[i] for i in range(len(beta_test))]
        alpha_test = 1440 * np.ones((len(beta_test)))     

        if n_unsat != 0:
            alpha_test[:n_unsat] = alpha_unsat[:n_unsat]

        low_rand_h = [-(b_S/100)*h_opt[i] for i in range(len(h_initial))]
        up_rand_h = [b_S/100*h_opt[i] for i in range(len(h_initial))]

        h_test = h_opt + np.random.uniform(low_rand_h,up_rand_h)
        c_t,lambda_t = Theorectical_dispersion_curve(c_min,c_max,c_step,lambda_OBS,n,alpha_test,beta_test,rho,h_test,delta_c)
        e_test = Misfit(c_t,c_OBS)

        store_all[0,w] = np.zeros((len(beta_test)))
        store_all[1,w] = np.zeros((len(h_test)))
        store_all[2,w] = np.zeros((len(alpha_test)))
        store_all[3,w] = np.zeros((len(c_t)))
        store_all[4,w] = np.zeros((len(lambda_t)))
        store_all[5,w] = np.zeros((len(e_test)))
        
        store_all[0,w] = beta_test
        store_all[1,w] = h_test
        store_all[2,w] = alpha_test
        store_all[3,w] = c_t
        store_all[4,w] = lambda_t
        store_all[5,w] = e_test 
        
        if e_test <= e_opt:
            e_opt = e_test
            print('Iteration # {} --- Misfit value {}%'.format(w,e_opt*100))            
            beta_opt = beta_test
            h_opt = h_test
            x_data.append(w)
            y_data.append(e_opt)

        if e_opt < e_max:
            Nend = w
            break

    if (up_low_boundary == 'Yes'):
        Vec = np.zeros((Nend,1))
        count = 0
        for i in range(Nend):
            temp = np.zeros((len(c_OBS_low),1))
            for j in range(len(c_OBS_low)):
                temp[j] = (c_OBS_low[j] < store_all[3,i][j] < c_OBS_up[j])
                if temp[j] == True: 
                    temp[j] = 1
                else:
                    temp[j] = 0
            if sum(temp) == len(c_OBS_low):
                Vec[i] = 1
                count += 1
        store_accepted = np.zeros((6,count))
        j = 0
        for i in range(Nend):
            if Vec[i] == 1:
                j += 1
                store_accepted[0,j] = np.zeros((len(beta_test)))
                store_accepted[1,j] = np.zeros((len(h_test)))
                store_accepted[2,j] = np.zeros((len(alpha_test)))
                store_accepted[3,j] = np.zeros((len(c_t)))
                store_accepted[4,j] = np.zeros((len(lambda_t)))
                store_accepted[5,j] = np.zeros((len(e_test)))
                store_accepted[0,j] = store_all[0,j]
                store_accepted[1,j] = store_all[1,j]
                store_accepted[2,j] = store_all[2,j]
                store_accepted[3,j] = store_all[3,j]
                store_accepted[4,j] = store_all[4,j]
                store_accepted[5,j] = store_all[5,j]
    else:
        store_accepted = np.NaN
    return store_all,store_accepted

store_all,store_accepted = Monte_Carlo_Analysis(c_min,c_max,c_step,delta_c,n,n_unsat,alpha_initial,nu_unsat,beta_initial,
                rho,h_initial,N_reversals,c_OBS,lambda_OBS,up_low_boundary,c_OBS_up,c_OBS_low,
                b_S,b_h,N_max,e_max)

MaxDepth = 16

def visualization(c_OBS,lambda_OBS,up_low_boundary,c_OBS_up,c_OBS_low,n,store_all,store_accepted,MaxDepth):
    NoPlot_all = len(store_all[0,:])
    store_e_all = np.zeros((NoPlot_all))

    for i in range(NoPlot_all):
        store_e_all[i] = store_all[5,i]

    sort_e_all_plot = np.sort(store_e_all)
    order_all_plot = np.argsort(store_e_all)
    sort_e_all_plot = sort_e_all_plot[::-1]
    order_all_plot = order_all_plot[::-1]
    fig,ax = plt.subplots(figsize=(4,6))
    plt.xlim(100,200)
    for j in range(NoPlot_all):
        c_plot = store_all[3,order_all_plot[j]]
        lambda_plot = store_all[4,order_all_plot[j]]
        ax.plot(c_plot,lambda_plot,linewidth = j/N_max)
    ax.set_xlabel('Phase velocity (m/s)')
    ax.set_ylabel('Wavelength (m)')
    plt.gca().invert_yaxis()
    plt.show()

visualization(c_OBS,lambda_OBS,up_low_boundary,c_OBS_up,c_OBS_low,n,store_all,store_accepted,MaxDepth)
