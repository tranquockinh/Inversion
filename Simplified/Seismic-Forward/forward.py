from data import *

class forward(object):
  
  def __init__(self):
    pass

  def boundaries(self,depth):
    # define boundaries
    z = sp.Symbol('z')
    lower = depth[:-1]
    upper = depth[1:]
    return z,lower,upper
  # def particle_disp_function(self):
  def weightAtLambda(self,lambdaScalar,depth):
      nLayer = len(depth)-1
      Atwavelen = lambdaScalar
      z,low,up = self.boundaries(depth)
      cv = np.array([0.2507, -0.4341, -0.8474*2*np.pi, -0.3933*2*np.pi])
      totA_term1 = (cv[0]/(cv[2]/Atwavelen))*(sp.exp(cv[2]/Atwavelen*np.inf)-sp.exp(cv[2]/Atwavelen*0))
      totA_term2 = (cv[1]/(cv[3]/Atwavelen))*(sp.exp(cv[3]/Atwavelen*np.inf)-sp.exp(cv[3]/Atwavelen*0))
      totA = totA_term1 + totA_term2
      unit = np.zeros((nLayer))
      for j in range(nLayer):
          unit_term1 = (cv[0]/(cv[2]/Atwavelen))*(sp.exp(cv[2]/Atwavelen*up[j])-sp.exp(cv[2]/Atwavelen*low[j]))
          unit_term2 = (cv[1]/(cv[3]/Atwavelen))*(sp.exp(cv[3]/Atwavelen*up[j])-sp.exp(cv[3]/Atwavelen*low[j]))
          unit[j] = unit_term1 + unit_term2
      weights = unit/totA
      return weights
    
  def Dispersion_curve(self,wavelen,depth,shear_wave_velocity):
    nLayer = len(depth)-1
    # Compute the weight matrix
    W = np.zeros((len(wavelen),nLayer))
    for i,lambdaValue in enumerate(wavelen):
        weights = self.weightAtLambda(lambdaValue,depth)
        W[i,:] = weights
    U,S,VT = np.linalg.svd(W,full_matrices=False)
    Matrix_inverse_W = np.dot(VT.T,np.dot(np.linalg.inv(np.diag(S)),U.T))
    # compute dispersion curve
    forward_S_wave = np.matmul(W,shear_wave_velocity)
    forward_R_wave = 0.93 * forward_S_wave
    return forward_S_wave,forward_R_wave

  def refineLayer(self,numLayer,coeff):      
      if (numLayer > len(wavelen)):
        numLayer = len(self.wavelen)
      inversion_wavelen = wavelen[:numLayer]
      layers = [0]
      for i in range(len(inversion_wavelen)):
          if i == (len(inversion_wavelen)-1): # index start from 0 because of the surface 
              layers.append(np.inf)
          else:
              layers.append(wavelen[:numLayer][i]*coeff)
      return inversion_wavelen,layers

f = forward()
forward_S_wave,forward_R_wave = f.Dispersion_curve(wavelen,depth,Swave)
inversion_wavelen,layers = f.refineLayer(n_inverted_layer,dwr)
forward_S_wave,forward_R_wave = f.Dispersion_curve(inversion_wavelen,layers,forward_S_wave)
