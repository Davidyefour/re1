import numpy as np
class get_fronzen_info():
    def __init__(self, fronzen_bits:np.ndarray):
        self.fronzen_bits = fronzen_bits
        self.N = len(fronzen_bits)
    
    def get_bit_layer(self):
        N = self.N
        layer_vec = np.zeros(N,dtype=np.int8)
        for phi in range(N):
            psi = np.floor(phi/2)
            layer = 0
            while np.mod(psi ,2) == 1:
                layer = layer + 1
                psi = np.floor(psi/2)
            layer_vec[phi] = layer
        return layer_vec

    def get_llr_layer(self, length = 0):
        if length == 0:
            N = self.N
        else:
            N = length
        layer_vec = np.zeros(N,dtype=np.int8)
        for phi in range(1,N):
            psi = phi
            layer = 0
            while np.mod(psi,2) == 0:
                psi = np.floor(psi/2)
                layer = layer + 1
            layer_vec[phi] = layer
        return layer_vec

    def get_lambda_offset(self):
        n = int(np.log2(self.N))
        lambda_offset = [2**i for i in range(n+1)]
        return lambda_offset

