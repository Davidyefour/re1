import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
ls = [1, 1.5, 2, 2.5, 3, 3.5]
# SCL DATA
# N128 =[0.3161,0.1831, 0.15501, 0.0331, 0.0018, 0.0006]
# N256 =[0.3081, 0.0881, 0.12001, 0.0111, 0.0011, 0.0005]
# N512 = [0.3021, 0.1121, 0.10501, 0.00501, 0.0006, 0.0003]
# N1024 = [0.2541, 0.0691, 0.0501,0.00121, 0.0004, 0.0002]
# N2048 = [0.2961, 0.0391, 0.0071, 0.0011, 0.0002, 0.0001]

# 2.0dB
CA2562 =np.array([125, 37, 8, 8, 7, 7])
CA2562 = CA2562 /1000
CA5122 =np.array([107, 37, 8, 2, 2, 1])
CA5122 = CA5122 / 1000
#1.5dB
CA2561 =np.array([323, 153, 75, 34, 21, 19])
CA2561 = CA2561 / 1000
CA5121 =np.array([303, 146, 65, 30, 19, 17])
L = [1, 2, 4, 8, 16, 32]
CA5121 = CA5121 / 1000
plt.axes(yscale='log')
plt.plot(L, CA2562, marker = '^')
plt.plot(L, CA5122, marker= 'v')
plt.plot(L, CA2561, marker='*')
plt.plot(L, CA5121, marker = 'o')
plt.grid(linestyle='--')
plt.xlabel("L")
plt.ylabel("FER")
plt.legend(["SNR=2.0 N=256","SNR=2.5 N=512","SNR=1.5 N=256","SNR=1.5 N=512"])
plt.show()
# scl vs ca-scl
'''
SCL4 = np.array([334, 120, 21, 1, 0.2, 0.15])
SCL4 =SCL4/1000
SCL8 = np.array([302, 112, 18, 1, 0.1, 0.08])
SCL8 =SCL8/1000
SCL16 = np.array([255, 111, 17, 2, 0.1, 0.07])
SCL16 = SCL16/1000

CA_SCL4 = np.array([260, 73, 12, 0.5, 0.041, 0.02])
CA_SCL4 = CA_SCL4/1000
CA_SCL8 = np.array([178, 42, 5, 0.2, 0.025, 0.012])
CA_SCL8 = CA_SCL8/1000
CA_SCL16 = np.array([139, 28, 3, 0.1, 0.015])
CA_SCL16 = CA_SCL16/1000


plt.axes(yscale='log')
plt.plot(ls, CA_SCL4, marker='*')
plt.plot(ls, CA_SCL8, marker='^')
plt.plot([1,1.5,2,2.5,3 ], CA_SCL16, marker='v')
plt.plot(ls, SCL4, marker='o')
plt.plot(ls, SCL8, marker='1')
plt.plot(ls, SCL16, marker = '3')
plt.legend(["CA_SCL4","CA_SCL8","CA_SCL16","SCL4","SCL8","SCL16"])
plt.grid(linestyle='--')
plt.xlabel("snr/dB")
plt.ylabel("FER")
plt.title("SCL vs CA-SCL")
plt.show()
'''