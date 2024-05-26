import json
from typing import Tuple
from pathlib import Path

import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

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
            print("\nPERMUTATION: FROM A_k:\n", A_k)
            A_k[[k, max_l]] = A_k[[max_l, k]]
            print("TO A_k':\n", A_k)

            print("\nPERMUTATION: FROM L:\n", L)
            L[[k, max_l]] = L[[max_l, k]]
            L[:, [k, max_l]] = L[:, [max_l, k]]
            print("TO L':\n", L)

            print("\nPERMUTATION: FROM P:\n", P)
            P[[k, max_l]] = P[[max_l, k]]
            print("TO P':\n", P)

            permutations_cnt += 1


        loginfo = f"k={k}, lead={A_k[k, k]}"
        print(f"\n{loginfo:=^40}")

        for i in range(k+1, n):

            mu = A_k[i, k] / A_k[k, k]
            L[i, k] = mu

            for j in range(k, n):
                # print(f"\ni={i}, j={j}, k={k}")

                # print(f"L[i][k] = {L[i, k]}, A[k][j] =  {A_k[k, j]}")
                # m = L[i, k]*A_k[k, j]
                # print(m, A_k[i, j])
                A_k[i, j] = np.round(A_k[i, j] - L[i, k]*A_k[k, j], 4)
                # print(A_k, '\n')
            
            loginfo = f"i={i}"
            print(f"\n{loginfo:-^40}")
            
            print(f"L = \n{L}")
            print(f"A_k = \n{A_k}")
        

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
            # print(f"z={z[j]}, L[i][j]={L[i, j]}")
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


def calculate_inverted_matrix(L: np.matrix, U: np.matrix, P: np.matrix) -> np.matrix:
    dim = L.shape
    n = dim[0]
    I = np.identity(n)
    A_i = np.zeros(dim)

    for i in range(n):
        x_i = solve_LU(L, U, P, I[:, i])
        A_i[:, i] = x_i

    # A_i = A_i.round(1)
    return A_i



def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        A = np.matrix(data["A"], dtype=np.float32)
        b = np.array(data["b"], dtype=np.float32)
    
    L, U, P, det = LUP_transfrom(A)

    print("\nTranform result")
    print("Source matrix A:\n", A)
    print("L:\n", L)
    print("U:\n", U)
    print("P:\n", P)
    print("L*U:\n", L.dot(U))

    x = solve_LU(L, U, P, b)
    print("Solution x:\n", x)

    n = A.shape[0]
    check_x = np.zeros(x.shape)

    for i in range(n):
        for j in range(n):
            check_x[i] += A[i, j]*x[j]
    
    print("Solution check: b\n", b, "\nwith x:\n", check_x)

    print("\ndet(A) =", round(det))
    print("NumPy det(A) =", np.linalg.det(A))
    
    A_i = calculate_inverted_matrix(L, U, P)
    print("\nInverted A matrix:\n", A_i)
    print("A * AI =\n", A.dot(A_i))



cur_dir = Path(__file__).parent
solve(cur_dir / "input3.json" )
