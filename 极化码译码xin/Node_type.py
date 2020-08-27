import numpy as np
from get_info_from_fronzen import get_fronzen_info
class Node_matrix():
    def __init__(self, frozen_bits):
        self.frozen_bits = 1 - frozen_bits
        self.len = len(self.frozen_bits)
    def getIMatrix(self): 
        '''
        根据传入的fronzen_bits，得到节点类型矩阵
        '''
        matrix = find(len(self.frozen_bits)).node_type_structure(self.frozen_bits)
        return matrix, self.get_psi_vec(matrix)

    def get_psi_vec(self, matrix):
        N = len(matrix)
        phi_vec = np.zeros(N, dtype=int)
        for i in range(N):
            psi = matrix[i,0]
            reduce_layer = int(np.log2(matrix[i,1]))
            psi = psi >> reduce_layer
            phi_vec[i] = psi & 1
        return phi_vec

class find():
    def __init__(self,len):
        self.len = len
        self.cnt = 0
        self.c_s = np.zeros((self.len, 3),dtype=np.int)

    def node_identifer(self ,f:np.array, z:list):
        # R0节点判定（0，0……，0）
        if np.all(f==0):
            self.c_s[self.cnt, 0] = z[0]
            self.c_s[self.cnt, 1] = len(f)
            self.c_s[self.cnt, 2] = 0
            self.cnt = self.cnt + 1
        
        # R1节点判定（1，1，……，1）
        elif np.all(f==1):
            self.c_s[self.cnt, 0] = z[0]
            self.c_s[self.cnt, 1] = len(f)    
            self.c_s[self.cnt, 2] = 1
            self.cnt = self.cnt + 1
        
        # R2节点判定（0，0，……，0，1）
        elif np.all(f[:-1]==0) and f[-1] == 1:
            self.c_s[self.cnt, 0] = z[0]
            self.c_s[self.cnt, 1] = len(f)    
            self.c_s[self.cnt, 2] = 2
            self.cnt = self.cnt + 1

        # R3节点判定(0,1,1……，1)        
        elif f[0]==0 and np.all(f[1:]):
            self.c_s[self.cnt, 0] = z[0]
            self.c_s[self.cnt, 1] = len(f)    
            self.c_s[self.cnt, 2] = 3
            self.cnt = self.cnt + 1
        
        else:
            half = int(len(f) / 2)
            self.node_identifer(f[:half], z[:half])
            self.node_identifer(f[half:], z[half:])

    def node_type_structure(self, fronzen_bits:np.array):
        self.node_identifer(fronzen_bits, [i for i in range(len(fronzen_bits))])
        row = 0
        while self.c_s[row,1] != 0:
            row = row + 1
        self.c_s = self.c_s[:row,:]
        return self.c_s

if __name__ == "__main__":
    fronzen_bits =  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1,
                                 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    f1 = 1- fronzen_bits
    print(f1)
    im = Node_matrix(fronzen_bits)
    matrix = im.getIMatrix()
    print(matrix)
    psi_vec = im.get_psi_vec(matrix)
    print(psi_vec)

