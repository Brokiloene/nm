import json
from typing import Tuple

import numpy as np

np.set_printoptions(precision=4, floatmode="fixed")

def LUP_transfrom(A) -> Tuple[np.matrix, np.matrix, np.matrix]:
    dim = A.shape
    n = dim[0]

    A_k = A.copy()
    L = np.diag(np.full(n, 1))
    P = np.identity(n)
    

    for k in range(n-1):
        lead = A_k[k, k]

        loginfo = f"k={k}, lead={lead}"
        print(f"{loginfo:=^40}")

        for i in range(k+1, n):
            
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



            mu = A_k[i, k] / lead
            L[i, k] = mu

            loginfo = f"i={i}"
            print(f"{loginfo:-^40}")
            
            print(f"L = \n{L}")

            for j in range(k, n):
                print(f"\ni={i}, j={j}, k={k}")

                print(f"L[i][k] = {L[i, k]}, A[k][j] =  {A_k[k, j]}")
                m = L[i, k]*A_k[k, j]
                print(m, A_k[i, j])
                A_k[i, j] = np.round(A_k[i, j] - L[i, k]*A_k[k, j], 4)
                print(A_k, '\n')

    for i in range(n):
        L[i, i] = 1.0
    return (L, A_k, P)


def solve_LU(L: np.matrix, U: np.matrix, b: np.array) -> np.array:
    # Lz = b
    # Ux = z

    n = L.shape[0]
    z = np.zeros(n)

    for i in range(n):
        s = 0
        for j in range(i):
            # print(f"z={z[j]}, L[i][j]={L[i, j]}")
            s += z[j]*L[i, j]
        z[i] = b[i] - s
    
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        s = 0
        for j in range(n-1, i, -1):
            s += x[j]*U[i, j]
        x[i] = (z[i] - s) / U[i, i]
    
    return x


def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        A = np.matrix(data["A"], dtype=np.float32)
        b = np.array(data["b"], dtype=np.float32)
    
    L, U, P = LUP_transfrom(A)

    print("Tranform result")
    print("Source matrix A:\n", A)
    print("L\n", L)
    print("U\n", U)
    print("L*U:\n", L.dot(U))

    x = solve_LU(L, U, b.dot(P))
    print("Solution x:\n", x)

    n = A.shape[0]
    check_x = np.zeros(x.shape)

    for i in range(n):
        for j in range(n):
            check_x[i] += A[i, j]*x[j]
    
    print("Solution check: b\n", b, "\nwith x:\n", check_x)
    



solve("input.json")
