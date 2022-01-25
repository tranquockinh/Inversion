
from forward import *
class backward():
    def __init__(self,wl,shearWave,thickness,coeff,test_nLayer):
        self.fw = forward(wl,shearWave,thickness)
        self.test_nLayer = test_nLayer
        fullSwave,fullRwave = self.fw.DC()
        fullThickness = self.fw.refineLayer(len(wl),coeff)[1:]
        self.coeff = coeff
        self.wl_test = wl[:test_nLayer]
        self.Swave_test = fullSwave[:test_nLayer]
        self.Rwave_test = fullRwave[:test_nLayer]
        self.thick_test = fullThickness[:test_nLayer]
        self.thick_test[-1] = np.inf

    def initialization(self):
        ini_weight = np.zeros((self.test_nLayer,self.test_nLayer))
        for i in range(self.test_nLayer):
            thickness = self.fw.refineLayer(i+1,self.coeff)[1:]
            ini_fw = forward(wl[:i+1],shearWave,thickness)
            W,invertW = ini_fw.weightMatrix()
            ini_weight[:i+1,:i+1] = W[::1,::1]
        self.init_Vs = np.matmul(ini_weight,self.Swave_test)
        return self.init_Vs,thickness

    def inversion(self):
        fw2 = forward(self.wl_test,self.init_Vs,self.thick_test)
        W,invertW = fw2.weightMatrix()
        self.iSwave = np.matmul(invertW,self.Swave_test)
        return self.iSwave

    def check_forward(self):
        checkFD = forward(wl,self.iSwave,self.thick_test)
        checkSwave,checkRwave = checkFD.DC()
        return checkSwave,checkRwave
    
    def parameters(self):
        test_Rwave = self.Rwave_test
        return test_Rwave
