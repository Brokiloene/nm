import json
from typing import Tuple
from pathlib import Path

import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def vector_norm(b: np.array, k: int):
    n = len(b)
    norm: np.matrix = np.sqrt( sum(b[i]**2 if i >= k else 0 for i in range(n)) )
    return norm.item()

def QR_decomposition(A: np.matrix) -> Tuple[np.matrix, np.matrix]:
    n = A.shape[0]
    A_k = A.copy()
    Q = np.eye(n)

    for k in range(n):
        v_k = np.zeros(n)
        for i in range(n):
            if i < k:
                continue
            if i == k:
                v_k[i] = A_k[i, k] + np.sign(A_k[i, k]) * vector_norm(A_k[:, k], k)
            else:
                v_k[i] = A_k[i, k]
        
        H = np.eye(n) - 2 * np.outer(v_k, v_k) / (v_k @ v_k)
        # print(H)
        
        A_k = H @ A_k
        Q @= H
    
    return Q, A_k

def solve_quadratic_equation(a, b, c) -> Tuple[np.single|np.csingle, np.single|np.csingle]:
    D = b**2 - 4*a*c

    λ_1 = (-b + np.emath.sqrt(D)) / (2*a)
    λ_2 = (-b - np.emath.sqrt(D)) / (2*a)
    return λ_1, λ_2


def QR_algorithm(A: np.matrix, EPS: float):
    n = A.shape[0]
    A_k = A.copy()

    converged = False
    num = 1
    while not converged:
        Q, R = QR_decomposition(A_k)

        print(f"===Iteration №{num}===")
        print("Q=\n",Q, "\nR=\n", R)
        print(f"A_{num}=\n", A_k, "\nQ*R=\n", Q@R)

        A_k = R @ Q
        num += 1
        for k in range(n):
            if vector_norm(A_k[:, k], k+1) > EPS:
                if vector_norm(A_k[:, k], k+2) > EPS:
                    break
        else:
            converged = True
    
    print(f"Final A_{num}=\n", A_k)

    λ = []
    k = 0
    while k < n:
        if vector_norm(A_k[:, k], k+1)*0.1 > EPS:
            λs = solve_quadratic_equation( 1, -(A_k[k,k] + A_k[k+1,k+1]), (A_k[k,k]*A_k[k+1,k+1] - A_k[k,k+1]*A_k[k+1,k]) )
            λ.extend(λs)
            k += 1
        else:
            λ.append(A_k[k,k])
        k += 1

    return np.array(λ)



def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        A = np.matrix(data["A"], dtype=np.float32)
        EPS = data["EPS"]

    eigenvalues = QR_algorithm(A, EPS)
    eigenvalues.sort()

    print("\nCalculated eignevalues using QR_algorithm: ", eigenvalues)

    numpy_eigenvalues = np.linalg.eig(A)[0]
    numpy_eigenvalues.sort()
    print("\nCalculated eignevalues using NumPy function:", numpy_eigenvalues)


cur_dir = Path(__file__).parent
solve(cur_dir / "input.json" )