import numpy as np
from crc import CRC_operation
from Node_type import Node_matrix
count = 0
class Decoder():
    def SC_nordecoder(self, llr:list, frozen_bits:list, K:int, lambda_offsets:list, llr_layer_vec:list, bit_layer_vec:list):
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
        PM = 0
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
                if P[0] > 0 :
                    PM = PM + P[0]
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
    
    def SCL_decoder_normal(self, llr:np.ndarray, frozen_bits:np.ndarray, K:int, L:int, lambda_offsets:np.ndarray,
                 llr_layer_vec:np.ndarray, bit_layer_vec:np.ndarray):
        '''
        llr : 接收端根据方差和接收到的数据算出的llr数组
        frozen_bits : 1代表冻结，0代表信息，index代表信道编号-1
        K : 信息位个数
        L : 译码器个数
        lamda_offset : 存放2^index，方便循环计算
        llr_layer_vec ： 计算LLR时候，从第几层开始
        bit_layer_vec : 计算完ui,返回到第几层
        info : 刚添加的
        '''
        # 计算模式设置
        f = f1 

        # 参数设置
        N = len(llr)
        n = int(np.log2(N))
        lazy_copy = np.zeros((n, L),dtype=int)

        # P[:,l] : 第l个译码器的llr
        P = np.zeros((N-1,L))
        
        # C[:,2*l] : 第l个译码器的中间比特
        C = np.zeros((2*N-1,2*L), dtype=int)
        
        
        # 路径度量
        PM = np.zeros(L)

        activepath = np.zeros(L, dtype=int)

        # 用来记录信息位的值
        cnt_u = 0
        u = np.zeros((K,L),dtype=np.int8)

        activepath[0] = 1
        
        lazy_copy[:,0] = 0
        # ------------------------- 主程序 -------------------------------
        for ui in range(N):
            layer = llr_layer_vec[ui]

            ui_mod_2 = ui % 2

            # ------------常规遍历译码器，得出ui的llr --------------------
            for l_index in range(L):
                # 如果译码器没被激活，下一个译码器
                if activepath[l_index] == 0:
                    continue
                
                # 解码的是左边第一个
                if ui == 0:
                    index1 = lambda_offsets[n-1]
                    # 赋初值给P[1]
                    for beta in range(index1):
                        P[beta+index1-1, l_index] = f(llr[beta], llr[beta+index1])                

                    # 向下更新P[1]
                    for i_layer in range(n-2,-1,-1):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer+1]
                        for beta in range(index1, index2):
                            P[beta-1, l_index] = f(P[beta+index1-1, l_index], P[beta+index2-1, l_index])
                
                # 解码的是右边第一个，初始化数组P，利用g函数
                elif ui == N/2:
                    index1 = lambda_offsets[n-1]
                    for beta in range(index1):
                        P[beta + index1 -1, l_index] = g(C[beta + index1 -1, 2 * l_index], llr[beta], llr[beta+index1])
                    
                    for i_layer in range(n-2,-1,-1):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer+1]
                        for beta in range(index1, index2):
                            P[beta-1, l_index] = f(P[beta+index1-1, l_index], P[beta+index2-1, l_index])

                # 其他情况
                else:
                    index1 = lambda_offsets[layer]
                    index2 = lambda_offsets[layer+1]
                    # 运行一次g，在递归运行f肯定可以算出该节点最左边的值
                    for beta in range(index1):
                        ctemp = C[beta + index1 -1, 2*l_index]
                        pleft =  P[beta + index2 - 1, lazy_copy[layer+1 ,l_index]]
                        pright = P[beta + index1 + index2 - 1,  lazy_copy[layer+1 ,l_index]]
                        P[beta + index1 - 1, l_index] = g(ctemp, pleft, pright) 

                    for i_layer in range(layer-1,-1,-1):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer + 1]
                        for beta in range(index1, index2):
                            P[beta-1, l_index] = f(P[beta+index1-1, l_index], P[beta+index2-1, l_index])
                
                # --------结束switch信息位的操作------------
            # ---------结束循环操作每个译码器得到ui的llr----------------

            # --------- PM权值更新相关操作 --------------
            # 信息为操作
            if frozen_bits[ui] == 0:
                # 用来储存路径度量， 对于一个路径L，PM（0，L）位ui等于0的路径度量，PM（1，L）位ui等于1的路径度量
                PM_pair = float('inf') * np.ones((2,L))
                for l_index in range(L):
                    if activepath[l_index] == 0:
                        continue
                    
                    # 由 p0（llr[ui]）决定是0还是1增加路径度量
                    if  P[0, l_index] < 0:
                        PM_pair[0, l_index] = PM[l_index] - P[0, l_index]
                        PM_pair[1, l_index] = PM[l_index]
                    else:
                        PM_pair[0, l_index] = PM[l_index] 
                        PM_pair[1, l_index] = PM[l_index] + P[0, l_index]

                # 排序路径度量，获得中值，原本的与中间值比较，小于的取1，大于取0。 1的用作下一次路径度量计算
                middle = min(L,2 * sum(activepath))
                PM_sort = np.sort(PM_pair.reshape(PM_pair.size))
                PM_judge = PM_sort[middle-1]
                compare = PM_pair <= PM_judge
                
                # 对于那种全false PM_pair[l_index],我们进行删除操作
                kill_count = 0
                kill = np.zeros(L,int)
                for l_index in range(L):
                    if not compare[0, l_index] and  not compare[1, l_index]:
                        kill[kill_count] = l_index
                        kill_count += 1
                
                # 根据PM_pair路径度量， 做出一定选择
                for l_index in range(L):
                    # 两个都满足， 从删除的PM取一个出来覆盖
                    if compare[0, l_index] and compare[1, l_index]:
                        # 将删除的PM弹出
                        kill_count -= 1
                        index = kill[kill_count]
                        activepath[index] = 1

                        # 懒复制
                        lazy_copy[:, index] = lazy_copy[:, l_index]
                        
                        # 将本身的值赋值给弹出的,并更新C和下一个u
                        u[:, index] = u[:, l_index]
                        u[cnt_u, l_index] = 0
                        u[cnt_u, index] = 1

                        C[0, 2 * l_index + ui_mod_2] = 0
                        C[0, 2 * index + ui_mod_2] = 1

                        # PM更新
                        PM[l_index] = PM_pair[0, l_index]
                        PM[index] = PM_pair[1, l_index]


                    
                    # PM是原来的值(ui=0)
                    elif compare[0, l_index] and not compare[1, l_index]:
                        u[cnt_u,l_index] = 0
                        C[0, 2*l_index + ui_mod_2] = 0
                        PM[l_index] = PM_pair[0, l_index]
                
                    # PM是原来的值(ui=1)
                    elif not compare[0, l_index] and compare[1, l_index]:
                        u[cnt_u,l_index] = 1
                        C[0, 2*l_index + ui_mod_2] = 1
                        PM[l_index] = PM_pair[1, l_index]

                cnt_u += 1

            # 信息为操作结束

            # 对冻结为操作, C中插入0和更新权值
            else:
                for l_index in range(L):
                    if activepath[l_index] == 0:
                        continue
                    
                    if P[0, l_index] < 0:
                        PM[l_index] -= P[0, l_index]
                
                    C[0, 2*l_index + ui_mod_2] = 0
        
            # --------- 结束更新PM权值操作 -----------

            # ---------- 更新C还有lazy_copy --------
            for l_index in range(L):
                if activepath[l_index] == 0:
                    continue
                if ui_mod_2 == 1:
                    bit_layer = bit_layer_vec[ui]
                    for i_layer in range(bit_layer):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer+1]
                        # 右    边 = 【lazy_copy左边 + 右边】 + 右边
                        for beta in range(index1, index2):
                            C[beta + index1 - 1, 2 * l_index + 1] =  \
                                np.mod(C[beta - 1, 2 * lazy_copy[i_layer, l_index] ] \
                                + C[beta - 1, 2 * l_index + 1], 2)
                            C[beta + index2 - 1, 2 * l_index + 1] = C[beta - 1, 2 * l_index + 1]
                        
                    index1 = lambda_offsets[bit_layer]
                    index2 = lambda_offsets[bit_layer + 1]
                    # 左边 = 【lazy_copy左边 + 右边】 + 右边
                    for beta in range(index1, index2):
                        C[beta + index1 - 1, 2 * l_index ] =  \
                            np.mod(C[beta - 1, 2 * lazy_copy[bit_layer, l_index] ] \
                                + C[beta - 1, 2 * l_index + 1], 2)
                        C[beta + index2 - 1, 2 * l_index ] = C[beta - 1, 2 * l_index +1] 
                    
                    # 更新C结束
                    
                
            #更新lazy_copy
            if ui < N-1:
                i_layer = llr_layer_vec[ui + 1] + 1
                for l_index in range(L):
                    if activepath[l_index]:
                        lazy_copy[:i_layer,l_index] = l_index

                # for i_layer in range(llr_layer_vec[ui+1] + 1):
                #     for l_index in range(L):
                #         if activepath[l_index]:
                #             lazy_copy[i_layer, l_index] = l_index
        # 所有ui判决结束

        PM_ascend = np.argsort(PM)

        return u[:,PM_ascend[0]]
        # 函数结束
    
    def CA_SCL_decoder(self, llr:np.ndarray, frozen_bits:np.ndarray, K:int, L:int, lambda_offsets:np.ndarray,
                 llr_layer_vec:np.ndarray, bit_layer_vec:np.ndarray, crc:np.ndarray):
        '''
        llr : 接收端根据方差和接收到的数据算出的llr数组
        frozen_bits : 1代表冻结，0代表信息，index代表信道编号-1
        K : 信息位个数
        L : 译码器个数
        lamda_offset : 存放2^index，方便循环计算
        llr_layer_vec ： 计算LLR时候，从第几层开始
        bit_layer_vec : 计算完ui,返回到第几层
        crc : 校验需要用到的多项式
        '''
        # 计算模式设置
        f = f1 

        # 参数设置
        N = len(llr)
        n = int(np.log2(N))
        lazy_copy = np.zeros((n, L),dtype=int)

        # P[:,l] : 第l个译码器的llr
        P = np.zeros((N-1,L))
        
        # C[:,2*l] : 第l个译码器的中间比特
        C = np.zeros((2*N-1,2*L), dtype=int)
        
        
        # 路径度量
        PM = np.zeros(L)

        activepath = np.zeros(L, dtype=int)

        # 用来记录信息位的值
        cnt_u = 0
        u = np.zeros((K,L),dtype=np.int8)

        activepath[0] = 1
        
        lazy_copy[:,0] = 0
        # ------------------------- 主程序 -------------------------------
        for ui in range(N):
            layer = llr_layer_vec[ui]

            ui_mod_2 = ui % 2

            # ------------常规遍历译码器，得出ui的llr --------------------
            for l_index in range(L):
                # 如果译码器没被激活，下一个译码器
                if activepath[l_index] == 0:
                    continue
                
                # 解码的是左边第一个
                if ui == 0:
                    index1 = lambda_offsets[n-1]
                    # 赋初值给P[1]
                    for beta in range(index1):
                        P[beta+index1-1, l_index] = f(llr[beta], llr[beta+index1])                

                    # 向下更新P[1]
                    for i_layer in range(n-2,-1,-1):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer+1]
                        for beta in range(index1, index2):
                            P[beta-1, l_index] = f(P[beta+index1-1, l_index], P[beta+index2-1, l_index])
                
                # 解码的是右边第一个，初始化数组P，利用g函数
                elif ui == N/2:
                    index1 = lambda_offsets[n-1]
                    for beta in range(index1):
                        P[beta + index1 -1, l_index] = g(C[beta + index1 -1, 2 * l_index], llr[beta], llr[beta+index1])
                    
                    for i_layer in range(n-2,-1,-1):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer+1]
                        for beta in range(index1, index2):
                            P[beta-1, l_index] = f(P[beta+index1-1, l_index], P[beta+index2-1, l_index])

                # 其他情况
                else:
                    index1 = lambda_offsets[layer]
                    index2 = lambda_offsets[layer+1]
                    # 运行一次g，在递归运行f肯定可以算出该节点最左边的值
                    for beta in range(index1):
                        ctemp = C[beta + index1 -1, 2*l_index]
                        pleft =  P[beta + index2 - 1, lazy_copy[layer+1 ,l_index]]
                        pright = P[beta + index1 + index2 - 1,  lazy_copy[layer+1 ,l_index]]
                        P[beta + index1 - 1, l_index] = g(ctemp, pleft, pright) 

                    for i_layer in range(layer-1,-1,-1):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer + 1]
                        for beta in range(index1, index2):
                            P[beta-1, l_index] = f(P[beta+index1-1, l_index], P[beta+index2-1, l_index])
                
                # --------结束switch信息位的操作------------
            # ---------结束循环操作每个译码器得到ui的llr----------------

            # --------- PM权值更新相关操作 --------------
            # 信息为操作
            if frozen_bits[ui] == 0:
                # 用来储存路径度量， 对于一个路径L，PM（0，L）位ui等于0的路径度量，PM（1，L）位ui等于1的路径度量
                PM_pair = float('inf') * np.ones((2,L))
                for l_index in range(L):
                    if activepath[l_index] == 0:
                        continue
                    
                    # 由 p0（llr[ui]）决定是0还是1增加路径度量
                    if  P[0, l_index] < 0:
                        PM_pair[0, l_index] = PM[l_index] - P[0, l_index]
                        PM_pair[1, l_index] = PM[l_index]
                    else:
                        PM_pair[0, l_index] = PM[l_index] 
                        PM_pair[1, l_index] = PM[l_index] + P[0, l_index]

                # 排序路径度量，获得中值，原本的与中间值比较，小于的取1，大于取0。 1的用作下一次路径度量计算
                middle = min(L,2 * sum(activepath))
                PM_sort = np.sort(PM_pair.reshape(PM_pair.size))
                PM_judge = PM_sort[middle-1]
                compare = PM_pair <= PM_judge
                
                # 对于那种全false PM_pair[l_index],我们进行删除操作
                kill_count = 0
                kill = np.zeros(L,int)
                for l_index in range(L):
                    if not compare[0, l_index] and  not compare[1, l_index]:
                        kill[kill_count] = l_index
                        kill_count += 1
                
                # 根据PM_pair路径度量， 做出一定选择
                for l_index in range(L):
                    # 两个都满足， 从删除的PM取一个出来覆盖
                    if compare[0, l_index] and compare[1, l_index]:
                        # 将删除的PM弹出
                        kill_count -= 1
                        index = kill[kill_count]
                        activepath[index] = 1

                        # 懒复制
                        lazy_copy[:, index] = lazy_copy[:, l_index]
                        
                        # 将本身的值赋值给弹出的,并更新C和下一个u
                        u[:, index] = u[:, l_index]
                        u[cnt_u, l_index] = 0
                        u[cnt_u, index] = 1

                        C[0, 2 * l_index + ui_mod_2] = 0
                        C[0, 2 * index + ui_mod_2] = 1

                        # PM更新
                        PM[l_index] = PM_pair[0, l_index]
                        PM[index] = PM_pair[1, l_index]


                    
                    # PM是原来的值(ui=0)
                    elif compare[0, l_index] and not compare[1, l_index]:
                        u[cnt_u,l_index] = 0
                        C[0, 2*l_index + ui_mod_2] = 0
                        PM[l_index] = PM_pair[0, l_index]
                
                    # PM是原来的值(ui=1)
                    elif not compare[0, l_index] and compare[1, l_index]:
                        u[cnt_u,l_index] = 1
                        C[0, 2*l_index + ui_mod_2] = 1
                        PM[l_index] = PM_pair[1, l_index]

                cnt_u += 1

            # 信息为操作结束

            # 对冻结为操作, C中插入0和更新权值
            else:
                for l_index in range(L):
                    if activepath[l_index] == 0:
                        continue
                    
                    if P[0, l_index] < 0:
                        PM[l_index] -= P[0, l_index]
                
                    C[0, 2*l_index + ui_mod_2] = 0
        
            # --------- 结束更新PM权值操作 -----------

            # ---------- 更新C还有lazy_copy --------
            for l_index in range(L):
                if activepath[l_index] == 0:
                    continue
                if ui_mod_2 == 1:
                    bit_layer = bit_layer_vec[ui]
                    for i_layer in range(bit_layer):
                        index1 = lambda_offsets[i_layer]
                        index2 = lambda_offsets[i_layer+1]
                        # 右    边 = 【lazy_copy左边 + 右边】 + 右边
                        for beta in range(index1, index2):
                            C[beta + index1 - 1, 2 * l_index + 1] =  \
                                np.mod(C[beta - 1, 2 * lazy_copy[i_layer, l_index] ] \
                                + C[beta - 1, 2 * l_index + 1], 2)
                            C[beta + index2 - 1, 2 * l_index + 1] = C[beta - 1, 2 * l_index + 1]
                        
                    index1 = lambda_offsets[bit_layer]
                    index2 = lambda_offsets[bit_layer + 1]
                    # 左边 = 【lazy_copy左边 + 右边】 + 右边
                    for beta in range(index1, index2):
                        C[beta + index1 - 1, 2 * l_index ] =  \
                            np.mod(C[beta - 1, 2 * lazy_copy[bit_layer, l_index] ] \
                                + C[beta - 1, 2 * l_index + 1], 2)
                        C[beta + index2 - 1, 2 * l_index ] = C[beta - 1, 2 * l_index +1] 
                    
                    # 更新C结束
                    
                
            #更新lazy_copy
            if ui < N-1:
                i_layer = llr_layer_vec[ui + 1] + 1
                for l_index in range(L):
                    if activepath[l_index]:
                        lazy_copy[:i_layer,l_index] = l_index

                # for i_layer in range(llr_layer_vec[ui+1] + 1):
                #     for l_index in range(L):
                #         if activepath[l_index]:
                #             lazy_copy[i_layer, l_index] = l_index
        # 所有ui判决结束

        PM_ascend = np.argsort(PM)

        crc_op = CRC_operation()
        for i in range(L):
            result = crc_op.crcDecry(u[:, PM_ascend[i]], crc)
            if sum(result) == 0:
                return u[:, PM_ascend[i]]

        return u[:, PM_ascend[0]]
        # 函数结束
    
    def fast_SCdecoder(self, llr,node_type_structure, lambda_offset, llr_layer_vec, psi_vec, bit_layer_vec, frozen_bits = None):
        f = f1
        N = len(llr)
        n = int(np.log2(N))
        C = np.zeros((2*N-1, 2), dtype=np.int)
        P = np.zeros((2*N-1))
        P[N - 1 : ] = llr
        node_size = len(node_type_structure)
        for i_node in range(node_size):
            # 获得该分块的长度
            M = node_type_structure[i_node,1]
            
            # 减少的层数，即遇到该节点不用继续往下
            reduce_layer = int(np.log2(M))

            # 根据分块开始位置，决定从第几层开始计算LLR
            llr_layer = llr_layer_vec[node_type_structure[i_node, 0]]

            # 根据分块，哪里结束中间比特计算
            bit_layer = bit_layer_vec[node_type_structure[i_node, 0] + M -1]

            # 用作bit递归
            psi = psi_vec[i_node]

            psi_mode_2 = psi%2

            if i_node == 0:
                for i_layer in range(n-1, reduce_layer-1, -1):
                    index1 = lambda_offset[i_layer] 
                    index2 = lambda_offset[i_layer+1]
                    for beta in range(index1, index2):
                        P[beta - 1] = f(P[beta + index1 - 1], P[beta + index2 - 1])
            else:
                index1 = lambda_offset[llr_layer]
                index2 = lambda_offset[llr_layer + 1]
                for beta in range(index1, index2):
                    P[beta - 1] = g(C[beta - 1,0], P[beta + index1 - 1], P[beta + index2 - 1])

                for i_layer in range(llr_layer-1, reduce_layer-1, -1):
                    index1 = lambda_offset[i_layer]
                    index2 = lambda_offset[i_layer + 1]
                    for beta in range(index1, index2):
                        P[beta - 1] = f(P[beta + index1 - 1], P[beta + index2 - 1])
            
            # 下面开始进行判决
            # 如果是R0节点
            if node_type_structure[i_node,2] == 0:
                for j in range(M-1, 2*M-1): 
                    C[j, psi_mode_2] = 0
            
            # 如果所示R1节点
            elif node_type_structure[i_node, 2] == 1:
                for j in range(M-1, 2*M-1):
                    C[j, psi_mode_2] = 1 if P[j] < 0 else 0
                    
            # R2节点（0，0，0，1）
            elif node_type_structure[i_node, 2] == 2:
                sumj = 0
                for j in range(M-1, 2*M-1):
                    sumj = sumj + P[j]

                repeat = 1 if sumj < 0 else 0

                for j in range(M-1, 2*M-1):
                    C[j, psi_mode_2] = repeat
            
            # R3节点（0，1，1，1）
            elif node_type_structure[i_node, 2] == 3:
                sub_code = np.zeros(M)
                for j in range(M-1, 2*M-1):
                    sub_code[j - M + 1] = P[j]
                x = sub_code < 0

                if np.mod(sum(x), 2) != 0:
                    sub_code = np.abs(sub_code)
                    min_index = np.argmin(sub_code)
                    x[min_index] = np.mod(x[min_index] + 1, 2)

                for j in range(M-1, 2*M-1): 
                    C[j, psi_mode_2] = x[j - M + 1]

            # end switch

            if psi_mode_2 == 1:
                for i_layer in range(reduce_layer, bit_layer):
                    index1 = lambda_offset[i_layer]
                    index2 = lambda_offset[i_layer + 1]                
                    for beta in range(index1, index2):
                        C[beta + index1 - 1, 1] = np.mod(C[beta - 1, 0] + C[beta - 1, 1], 2)
                        C[beta + index2 - 1, 1] = C[beta - 1, 1]

                # 左边节点返回，用来计算父节点右边LLR
                index1 = lambda_offset[bit_layer]
                index2 = lambda_offset[bit_layer + 1]
                for beta in range(index1, index2):
                    C[beta + index1 - 1, 0] = np.mod(C[beta - 1, 0] + C[beta - 1, 1], 2)
                    C[beta + index2 - 1, 0] = C[beta - 1, 1]
        # end for i_node
        result = C[N-1: , 0]
        result = transXtoU(result)
        if frozen_bits is  None:
            return result
            
        else:
            return transToInfo(result, frozen_bits)
    
    def Prefast_SCLdecoder(self, llr, frozen_bits, lambda_offset, llr_layer_vec, K, L, bit_layer_vec, crc,node_type_structure=None, psi_vec = None):
        if node_type_structure is None:
            matrix = Node_matrix(frozen_bits)
            node_type_structure , psi_vec= matrix.getIMatrix()
        
        # fast-sc 解码
        poloar_info = self.fast_SCdecoder(llr, node_type_structure, lambda_offset, llr_layer_vec, psi_vec, bit_layer_vec, frozen_bits)
        # 对解码结果进行crc校验
        crc_op = CRC_operation()
        check = crc_op.crcDecry(poloar_info, crc)
        if sum(check) == 0:
            return poloar_info
        
        else:
            return self.CA_SCL_decoder(llr, frozen_bits, K, L, lambda_offset, llr_layer_vec, bit_layer_vec, crc)
        

        
def f1(a, b):
    return  np.sign(a) * np.sign(b) * min(abs(a),abs(b))

def g(u, a, b):
    return (1-2*u) * a + b

def transXtoU(x):
    length = len(x)
    u = np.zeros(length)
    index = [i for i in range(length)]
    def tans(x,i):
        n = len(x)
        half = int(n/2)
        if n == 1:
            u[i[0]] = x[0]
        else:
            x[:half] = np.mod(x[:half] + x[half:], 2) 
            tans(x[:half],i[:half])
            tans(x[half:], i[half:])
    tans(x, index)
    return u 

def transToInfo(x, fronzen_bits):
    '''
    x : fast_SCdecoder return x
    fronzen_bits : the frozen_bits of fast_SCdecoder
    '''
    infoN = int(sum(fronzen_bits))
    info = np.zeros(infoN, dtype=int)
    index = 0
    for i in range(len(fronzen_bits)):
        if fronzen_bits[i] == 0:
            info[index] = x[i]
            index += 1
    return info

if __name__ == "__main__":
    f1(1,2)
    f1(1,2)
    f1(1,2)
    print(count)

