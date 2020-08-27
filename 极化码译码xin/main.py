import numpy as np
from get_identifer_matrix import identifer_matrix
from fastSCdecoder import *
# from SCL_decoder import * 

fronzen_bits =  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
                            0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
im = identifer_matrix(fronzen_bits)
matrix = im.getIMatrix()
psi_vec = im.get_psi_vec(matrix)
bit_layer_vec = im.get_bit_layer()
llr_layer_vec = im.get_llr_layer()
llr = np.array([8.2, 12.8, -1.7, 9.4, 7.5, 1.7, 4.8, -5.1, 19.1, 3.5, -11.1, 17.1,
        -3.7, 6.1, 8.9, 5.6, 5.9, -1.0, -1.3, 11.4, -3.9, 2.0, 8.9, 12.1, 8.1, 10, 8.9, 
        5.2, 7.4, 3.5, 9.5, -10.4])


lambda_offset = im.get_lambda_offset()
x = fast_SCdecoder(llr, matrix, lambda_offset, llr_layer_vec, psi_vec, bit_layer_vec,fronzen_bits)
x = x.astype(int)
print("发送端发送消息：", x)
# SCL_decoder_normal(llr, fronzen_bits,8,8,lambda_offset,llr_layer_vec,bit_layer_vec)
