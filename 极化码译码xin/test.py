import numpy as np
import time
import math
from get_info_from_fronzen import get_fronzen_info
from polar_encode import polar_ecode
import matplotlib.pyplot as plt
from crc import CRC_operation
from polar_Decoder import Decoder
from Node_type import Node_matrix
from multiprocessing import process

for code_N in [256, 512]:
    N = code_N
    test_snr = [1.5, 2, 2.5]
    # design_L = [8,16]
    R = 0.5
    K = int(R*N)

    crc_len = 5
    # design_snr = 2.5
    for design_snr in test_snr:

        # -------------- 固定部分 ----------------
        crc_op = CRC_operation()
        crc = crc_op.get_g(crc_len)
        # 得到信道可靠性排序
        filename = 'channel/' + str(N) + '_' + str(design_snr) + '.txt'
        channel = np.loadtxt(filename,dtype=int)
        channel = channel -1 # 数组下标从0开始
        info_channel = channel[:K] # info_channel里面的数据为：[127 125 123………………84 30]这样的可选信道
        info_channel = np.sort(info_channel) # 排序info_channel，信道从小到大（30，31，50……）
        sigma = 1/math.sqrt(2 * R) * 10**(-design_snr/20)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_channel] = 0 # info_channel都是信息位，相应的冻结为置0

        # ------------- 测试部分参数 ---------------
        fast_SC_rate = 0
        SC_rate = 0
        CA_rate = 0
        new_rate = 0

        new_time = 0
        CA_time = 0
        SC_time = 0
        fast_SC_time = 0

        # 快速SC译码准备
        node_matrix = Node_matrix(frozen_bits)
        node_type_structure, psi_vec = node_matrix.getIMatrix()
        # --------------- 测试随机用例 -----------------
        block = 1000 # 用例数量
        for i in range(block):
            # 生成随机数据
            info = np.random.rand(K-crc_len)
            info[info<0.5] = 0
            info[info >=0.5] = 1
            info = info.astype(int)
            
            # 添加crc校验
            with_crc = crc_op.crcEncry(info, crc)
            info_with_crc = np.append(info, with_crc)

            u = np.zeros(N, dtype=int) 
            u[info_channel] = info_with_crc # 放入消息位置中
            x = polar_ecode(u) # 极化码编码
            bpsk = 1 - 2 * x # 模拟bpsk调制

            noise = sigma * np.random.randn(N)
            y = bpsk + noise
            llr = 2/pow(sigma,2) * y




            parm_info = get_fronzen_info(frozen_bits)
            bit_layer_vec = parm_info.get_bit_layer()
            llr_layer_vec = parm_info.get_llr_layer()
            lambda_offset = parm_info.get_lambda_offset()

            # 解码部分
            L = 8
            decoder = Decoder()

            # 新方法测试
            t_start = time.time()
            polar_info_with_crc = decoder.Prefast_SCLdecoder(llr, frozen_bits, lambda_offset, llr_layer_vec, K, L ,bit_layer_vec, crc, node_type_structure, psi_vec)
            new_time = new_time + time.time() - t_start
            
            polar_info = polar_info_with_crc[:K-crc_len]
            err = any(polar_info - info)
            if err:
                new_rate += 1

            # CA方法测试
            t_start = time.time()
            polar_info_with_crc = decoder.CA_SCL_decoder(llr, frozen_bits, K, L, lambda_offset, llr_layer_vec, bit_layer_vec, crc)
            CA_time = CA_time + time.time() - t_start
            
            polar_info = polar_info_with_crc[:K-crc_len]
            err = any(polar_info - info)
            if err:
                CA_rate += 1

            # SC方法测试
            t_start = time.time()
            polar_info = decoder.SC_nordecoder(llr, frozen_bits, K, lambda_offset, llr_layer_vec, bit_layer_vec)
            SC_time = SC_time + time.time() - t_start
            
            err = any(polar_info - info_with_crc)
            if err:
                SC_rate += 1
            

            # fast_sc方法测试
            t_start = time.time()
            polar_info = decoder.fast_SCdecoder(llr, node_type_structure, lambda_offset, llr_layer_vec, psi_vec, bit_layer_vec, frozen_bits)
            fast_SC_time = fast_SC_time + time.time() - t_start
            
            err = any(polar_info - info_with_crc)
            if err:
                fast_SC_rate += 1

        print("码长：", code_N, " 信噪： ", design_snr)
        print("NEW, CA, SC, FAST-SC误帧率为：",new_rate, CA_rate, SC_rate, fast_SC_rate)
        print("NEW, CA, SC, FAST-SC时间为：", new_time, CA_time, SC_time, fast_SC_time)
        print('\n\n')