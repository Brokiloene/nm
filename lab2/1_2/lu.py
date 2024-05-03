from typing import Tuple

import numpy as np


def LUP_transfrom(A) -> Tuple[np.matrix, np.matrix, np.matrix, float]:
    dim = A.shape
    n = dim[0]

    A_k = A.copy()
    L = np.diag(np.full(n, 1.0, dtype=np.float32))
    P = np.identity(n)
    permutations_cnt = 0

    for k in range(n-1):

        row = A_k[k, k]
        max_l = k

        for l in range(k+1, n):
            if abs(row) < abs(A_k[l, k]):
                row = abs(A_k[l, k])
                max_l = l
        
        if max_l != k:
            A_k[[k, max_l]] = A_k[[max_l, k]]

            L[[k, max_l]] = L[[max_l, k]]
            L[:, [k, max_l]] = L[:, [max_l, k]]

            P[[k, max_l]] = P[[max_l, k]]

            permutations_cnt += 1

        for i in range(k+1, n):

            mu = A_k[i, k] / A_k[k, k]
            L[i, k] = mu

            for j in range(k, n):
                A_k[i, j] = np.round(A_k[i, j] - L[i, k]*A_k[k, j], 4)

        u_cumprod = 1

        for i in range(n):
            u_cumprod *= A_k[i, i]
        
        det = u_cumprod * pow(-1, permutations_cnt)
    

    return (L, A_k, P, det)


def solve_LU(L: np.matrix, U: np.matrix, P: np.matrix, b: np.array) -> np.array:
    b = P.dot(b)
    n = L.shape[0]

    # Lz = b
    z = np.zeros(n)

    for i in range(n):
        s = 0
        for j in range(i):
            s += z[j]*L[i, j]
        z[i] = b[i] - s
    
    # Ux = z
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        s = 0
        for j in range(n-1, i, -1):
            s += x[j]*U[i, j]
        x[i] = (z[i] - s) / U[i, i]
    
    return x

def solve_SLAE(A: np.matrix, b: np.array) -> np.array:
    L, U, P, det = LUP_transfrom(A)
    x = solve_LU(L, U, P, b)
    return x
