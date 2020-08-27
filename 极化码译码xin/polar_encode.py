import numpy as np
def polar_ecode(u):
    N = len(u)
    half = int(N/2)
    if N == 1:
        return u
    else:
        u1u2 = np.mod(u[:half] + u[half:], 2)
        u2 = u[half:]
        return np.concatenate((polar_ecode(u1u2), polar_ecode(u2)), axis=0)

def polar_encode(u, lambda_offset, llr_layer_vec):
    pass    

if __name__ == "__main__":
    u =np.array([0,1,0,1,0,1,0,1]) 
    x = polar_ecode(u)
    print(x)