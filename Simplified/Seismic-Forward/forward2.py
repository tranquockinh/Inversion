from data import *

class forward_engine(object):
    
    def __init__(self,depth,shear_wave,wavelen):
        self.depth = depth
        self.nunber_layers = len(depth) - 1
        self.number_lambda = len(wavelen)
        self.shear_wave_velocity = shear_wave
        self.wavelen = wavelen

    def weight_atLambda(self,atLambda):

        # define boundaries
        z = sp.Symbol('z')
        low = self.depth[:-1]
        up = self.depth[1:]

        cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi, -0.3933*2*np.pi])

        totA_term1 = (cv[0]/(cv[2]/atLambda))*(sp.exp(cv[2]/atLambda*np.inf)-sp.exp(cv[2]/atLambda*0))
        totA_term2 = (cv[1]/(cv[3]/atLambda))*(sp.exp(cv[3]/atLambda*np.inf)-sp.exp(cv[3]/atLambda*0))
        totA = totA_term1 + totA_term2

        unit = np.zeros((self.nunber_layers))

        for j in range(self.nunber_layers):
            unit_term1 = (cv[0]/(cv[2]/atLambda))*(sp.exp(cv[2]/atLambda*up[j])-sp.exp(cv[2]/atLambda*low[j]))
            unit_term2 = (cv[1]/(cv[3]/atLambda))*(sp.exp(cv[3]/atLambda*up[j])-sp.exp(cv[3]/atLambda*low[j]))
            unit[j] = unit_term1 + unit_term2
        weights = unit/totA

        return weights
    
    def weight_allLambda(self):
        rows = self.number_lambda
        cols = self.nunber_layers
        weightMatrix = np.zeros((rows,cols))
        for idx,thisLambda in enumerate(self.wavelen):
            weights_thisLambda = self.weight_atLambda(thisLambda)
            weightMatrix[idx,:] = weights_thisLambda
        return weightMatrix
    
    def sigular_value_de(self,weightMatrix):
        # weightMatrix = self.weight_allLambda()
        U,S,VT = np.linalg.svd(weightMatrix,full_matrices=False)
        invert_W = np.dot(VT.T,np.dot(np.linalg.inv(np.diag(S)),U.T))
        return U,S,VT,invert_W

    def weightTable(self,index ='Lambda {}',columns='Layer {}',Name = 'Table name'):
        weightMatrix = self.weight_allLambda()
        lambdaLabels = []
        layerLabels = []
        for i in range(len(weightMatrix[:,0])):
            lambdaLabels.append(index.format(i+1))
        for j in range(len(weightMatrix[0,:])):
            layerLabels.append(columns.format(j+1))
        df = DataFrame(weightMatrix,index=Index(lambdaLabels, name=Name),columns=layerLabels)
        df.style

    def dispersion_curve(self):
        S_wave_velo_forward = np.matmul(self.weight_allLambda(),self.shear_wave_velocity) 
        R_wave_velo_forward = spr * S_wave_velo_forward
        return S_wave_velo_forward,R_wave_velo_forward

class backward_engine(forward_engine):

    def __init__(self,number_invert_layers):
        forward_engine.__init__(self, depth, Swave, wavelen)
        self.S_wave_velo_forward,self.R_wave_velo_forward = forward_engine.dispersion_curve(self)
        self.full_wavelegnth = wavelen
        self.number_invert_layers = number_invert_layers
        self.shearwave_for_inversion = self.S_wave_velo_forward[:self.number_invert_layers]
        
    def inversion_data(self):
        # 1. wavelength
        wavelen_for_inversion = self.full_wavelegnth[:self.number_invert_layers]
        # 2. depth
        temp = dwr * wavelen_for_inversion[:self.number_invert_layers]
        depth_for_inversion = np.append(0,temp)
        depth_for_inversion[-1] = np.inf
        # 3.shear wave
        
        Rwave_for_check = self.R_wave_velo_forward[:self.number_invert_layers]
        return depth_for_inversion

    def weight_allLambda_inversion(self):
        # inversion input
        depth_for_inversion = self.inversion_data()
        # Compute the weight matrix with inversion input
        self.forward_modeling = forward_engine(depth_for_inversion, self.shearwave_for_inversion, self.full_wavelegnth)
        weight_matrix_full = self.forward_modeling.weight_allLambda()
        self.weight_for_inversion = weight_matrix_full[:self.number_invert_layers,:self.number_invert_layers]
        # visualize the weight matrix in table
        weight_for_inversion_table = self.forward_modeling.weightTable()

    def inversion(self):
        self.weight_allLambda_inversion()
        S_wave_inversion = np.matmul(np.linalg.inv(self.weight_for_inversion),self.shearwave_for_inversion)
        return S_wave_inversion
    
    # Eigen-values analysis
    def analysis_SVD(self):
        # weight_for_inversion = self.weight_allLambda_inversion()
        U,S,VT,invert_W = self.forward_modeling.sigular_value_de(self.weight_for_inversion)
        return U,np.diag(S),VT,invert_W

    def echelon(self):

        Ab = np.concatenate((self.weight_for_inversion,self.shearwave_for_inversion[:,np.newaxis]),axis=1)
        Matrix_Ab = sp.Matrix(Ab)
        rref_Matrix_Ab = Matrix_Ab.rref()
        print(rref_Matrix_Ab)
        return rref_Matrix_Ab
        
    def check_dispersion_curve(self):
        depth_for_inversion = self.inversion_data()
        S_wave_inversion = self.inversion()
        check_forward_engine = forward_engine(depth_for_inversion, S_wave_inversion, self.full_wavelegnth)
        # check_weight_matrix = check_forward_engine.weight_allLambda()
        check_S_wave_velo_forward,check_R_wave_velo_forward = check_forward_engine.dispersion_curve()
        return check_S_wave_velo_forward,check_R_wave_velo_forward

forward_model = forward_engine(depth,Swave,wavelen)
S_wave_velo_forward,R_wave_velo_forward = forward_model.dispersion_curve()
print(R_wave_velo_forward)

backward_model = backward_engine(3)
# inversion results
S_wave_inversion = backward_model.inversion()
print(S_wave_inversion)

# see what insides
U,S,VT,invert_W = backward_model.analysis_SVD()

# Lood the row reduced ECHELON form (reff)
# backward_model.echelon()

# check dispersion curve
check_S_wave_velo_forward,check_R_wave_velo_forward = backward_model.check_dispersion_curve()
print(check_R_wave_velo_forward)

