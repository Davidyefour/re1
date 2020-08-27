import numpy as np
class CRC_operation():
    def crcEncry(self, data, crc):
        '''
        data ： 需要加密的数据
        crc : 约定不可分解多项式
        return ：余数
        '''
        n = len(crc)
        m = len(data)
        arr = np.append(data, np.zeros(n))
        arr = arr.astype(int)
        for i in range(m+1):
            if arr[i] == 1:
                arr[i:i+n] = np.bitwise_xor(arr[i:i+n], crc)
        return arr[-n:]
    
    def crcDecry(self, data, crc):
        '''
        data ： 需要加密的数据
        crc : 约定不可分解多项式
        return ：余数
        '''
        n = len(crc)
        m = len(data)
        arr = np.append(data, np.zeros(n))
        arr = arr.astype(int)
        for i in range(m+1):
            if arr[i] == 1:
                arr[i:i+n] = np.bitwise_xor(arr[i:i+n], crc)
        return arr[-n:]

    def XOR(self, ls1, ls2):
        return np.mod(ls1 + ls2,2)

    def get_g(self, crc_len:int):
        if crc_len == 5:
            g = np.array([1, 0, 0, 1, 1])

        elif crc_len == 7:
            g = np.array([1, 0, 0, 0, 0, 1, 1])

        elif crc_len == 13:
            g = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1])
        
        elif crc_len == 17:
            g = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
        
        elif crc_len == 25:
            g = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
            g = g.astype(int)

        else:
            raise Exception("没有该长度的crc 现在紧支持[5, 7, 13, 17, 25]")
        return g

if __name__ == "__main__":  
    test = 0
    for i in range(500000):
        k = np.random.rand(15)
        k[k>0.5] = 1
        k[k<=0.5] = 0
        k = k.astype(int)
        
        crc_op = CRC_operation()
        mycrc = crc_op.get_g(17)
        check = crc_op.crcEncry(k, mycrc)
        if sum(check) == 0:
            test += 1
    print(test)