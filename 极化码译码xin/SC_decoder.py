import numpy as np
import math
import matplotlib.pyplot as plt
import time
def SC_nordecoder(llr:list, frozen_bits:list, K:int, lambda_offsets:list, llr_layer_vec:list, bit_layer_vec:list):
    '''
    llr : 接收端根据方差和接收到的数据算出的llr数组
    frozen_bits : 1代表冻结，0代表信息，index代表信道编号-1
    K 信息位个数
    lamda_offset : 存放2^index，方便循环计算
    llr_layer_vec ： 计算LLR时候，从第几层开始
    bit_layer_vec : 计算完ui,返回到第几层
    '''
    # 计算模式设置
    f = f1 

    # 参数设置
    N = len(llr)
    n = int(np.log2(N))
    P = np.zeros(N-1)
    C = np.zeros((2*N-1,2), dtype=np.int8)
    u = np.zeros(K,dtype=np.int8)
    u_count = 0 # 用来保存信息位
    # -------- 主程序 -------------
    # 一次解码N各比特
    for ui in range(N):
        # 解码的是第一个比特，初始化LLR数组P，循环向下，算出u1，LLR
        if(ui==0):
            index1 = lambda_offsets[n-1]
            for beta in range(index1):
                P[beta+index1-1] = f(llr[beta], llr[beta+index1])
            
            
            for i_layer in range(n-2,-1,-1):
                index1 = lambda_offsets[i_layer]
                index2 = lambda_offsets[i_layer+1]
                for beta in range(index1, index2):
                    P[beta-1] = f(P[beta+index1-1], P[beta+index2-1])

        # 解码的是右边第一个，初始化数组P，利用g函数
        elif(ui==N/2):
            index1 = lambda_offsets[n-1]
            for beta in range(index1):
                P[beta + index1 -1] = g(C[beta + index1 -1,0], llr[beta], llr[beta+index1])
            
            for i_layer in range(n-2,-1,-1):
                index1 = lambda_offsets[i_layer]
                index2 = lambda_offsets[i_layer+1]
                for beta in range(index1, index2):
                    P[beta-1] = f(P[beta+index1-1], P[beta+index2-1])
        
        # 其他节点计算方式
        else:
            llr_layer = llr_layer_vec[ui]
            index1 = lambda_offsets[llr_layer]
            index2 = lambda_offsets[llr_layer+1]
            # 运行一次g肯定可以算出该节点最左边的值
            for beta in range(index1,index2):
                P[beta - 1] = g(C[beta - 1,0], P[beta + index1 - 1], P[beta + index2 - 1])
            
            
            for i_layer in range(llr_layer,0,-1):
                index1 = lambda_offsets[i_layer-1]
                index2 = lambda_offsets[i_layer]
                for beta in range(index1, index2):
                    P[beta-1] = f(P[beta+index1-1], P[beta+index2-1])
        
        # 经过上面一堆东西，可以得到ui的LLR，进行判决
        ui_mod2 = ui%2

        if frozen_bits[ui] == 1: # 冻结位
            C[0, ui_mod2] = 0
        else:
            if P[0] < 0:
                C[0, ui_mod2] = 1
            else:
                C[0, ui_mod2] = 0
            u[u_count] = C[0, ui_mod2]
            u_count = u_count+1

        # 如果是右节点，开始处理C数组 
        if ui_mod2 == 1:
            bit_layer = bit_layer_vec[ui]
            # 递归更新右边的值，每次可能计算多或0层右边，必须有一层左边
            for i_layer in range(bit_layer):
                index1 = lambda_offsets[i_layer]
                index2 = lambda_offsets[i_layer + 1]
                for beta in range(index1, index2):
                    C[beta + index1 - 1, 1] = np.mod(C[beta - 1, 0] + C[beta - 1, 1], 2)
                    C[beta + index2 - 1, 1] = C[beta - 1, 1]

            # 看是否有左边节点返回，用来计算右边LLR
            index1 = lambda_offsets[bit_layer]
            index2 = lambda_offsets[bit_layer + 1]
            for beta in range(index1, index2):
                C[beta + index1 - 1, 0] = np.mod(C[beta - 1, 0] + C[beta - 1, 1], 2)
                C[beta + index2 - 1, 0] = C[beta - 1, 1]
    return u

def f1(a, b):
    return  np.sign(a) * np.sign(b) * min(abs(a),abs(b))

def g(u, a, b):
    return (1-2*u) * a + b

def get_bit_layer(N:int):
    layer_vec = np.zeros(N,dtype=np.int8)
    for phi in range(N):
        psi = np.floor(phi/2)
        layer = 0
        while np.mod(psi ,2) == 1:
            layer = layer + 1
            psi = np.floor(psi/2)
        layer_vec[phi] = layer
    return layer_vec

def get_llr_layer(N:int):
    layer_vec = np.zeros(N,dtype=np.int8)
    for phi in range(1,N):
        psi = phi
        layer = 0
        while np.mod(psi,2) == 0:
            psi = np.floor(psi/2)
            layer = layer + 1
        layer_vec[phi] = layer
    return layer_vec

def RN(n):
    '''
    转置矩阵，其功能为将u=(u1,u2,u3,u4) -> (u1,u3,u2,u4)
    '''
    k = int(n/2)
    r = np.zeros(shape=(n,n))
    for i in range(k):
        r[2*i,i] = 1
        r[2*i+1,k+i] = 1
    return r


def BN(n):
    '''
    其递推公式为 Bn = Rn(I2*Bn/2))
    '''
    if n == 2:
        return np.eye(2)
    else:
        return np.matmul(RN(n),np.kron(np.eye(2),BN(int(n/2) ) ) ) 



def GN(n):
    '''
    生成矩阵 GN = BN * F(*)n
    '''
    F = np.array([[1,0],[1,1]])
    temp = F
    N = int(math.log(n,2))
    for i in range(1,N):
        temp = np.kron(F,temp)
    return np.matmul(temp,BN(n))   


def printScatter(index = 10):
    n = [2**i for i in range(1,index+1)]

    W = np.zeros(shape =(n[index-1],n[index-1]))

    W[0,0] = 0.5

    for i in n:
        for j in range(0,int(i/2)):
            W[i-1,2*j] = W[int(i/2)-1, j] ** 2
            W[i-1, 2*j+1] = 2 * W[int(i/2)-1, j] - W[int(i/2)-1, j]**2
    Wls = [i for i in W[2**index-1]]   
    return np.array(Wls)

def findChannel(index = 10):
    n = [2**i for i in range(1,index+1)]

    W = np.zeros(shape =(n[index-1],n[index-1]))

    W[0,0] = 0.5

    for i in n:
        for j in range(0,int(i/2)):
            W[i-1,2*j] = W[int(i/2)-1, j] ** 2
            W[i-1, 2*j+1] = 2 * W[int(i/2)-1, j] - W[int(i/2)-1, j]**2
    Wls = [i for i in W[2**index-1]]   
    return Wls

if __name__ == "__main__":

    situation = '''0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 
    0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 
    0 1 0 1 1 1 0 0 0 0 0 0 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 
    1 1 0 1 1 1 1 1 1 1 0 0 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    '''
    # filename = r'D:\vscode\python\极化码译码\fronzen_bits\256bits.txt'
    # ls = np.loadtxt(filename, dtype=int)
    # mat = identifer_matrix(ls).getIMatrix()
    # print(mat)
    index = 12
    w = printScatter(index)
    # ls = [i for i in range(2**index)]
    plt.xlabel("channel index")
    plt.ylabel("I(W)")    
    # for i in ls: 
    #     if w[i] < 0.01:
    #         plt.scatter(i, w[i], marker='*', s=1, color='orange')
    #     elif w[i] > 0.99:
    #         plt.scatter(i, w[i], marker='*', s=1, color='blue')
    #     else:
    #         plt.scatter(i, w[i], marker='*', s=1, color='red')
    # plt.show()

    w[w > 0.99] = 1
    w[w < 0.01] = 0
    w[w<1] = 0
    print(sum(w))
