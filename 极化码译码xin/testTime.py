from multiprocessing import Process
import time
def waitting(timing):
    t = time.time()
    while t - timing < 10:
        t = time.time()
    print("函数执行完毕")


if __name__ == "__main__":
    t = time.time()
    pls = []
    for i in range(3):
        p = Process(target= waitting, args=[t,])
        pls.append(p)
    for ps in pls:
        ps.start()
        print("函数开始")
