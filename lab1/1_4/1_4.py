import json
from typing import Tuple
from pathlib import Path

import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def matrix_norm(A: np.matrix) -> float:
    n = A.shape[0]
    res = np.sqrt( sum(A[i,j]**2 if i < j else 0 for i in range(n) for j in range(n)) )
    return res


def rotation_method(A: np.matrix, EPS) -> Tuple[np.array, np.matrix]:
    n = A.shape[0]

    converged = False
    num = 1
    U = np.eye(n)
    while not converged:
        cur_max = 0
        l, k = 0, 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.abs(A[i, j]) > abs(cur_max):
                    cur_max = A[i, j]
                    l = i
                    k = j

        phi = 0.5 * np.arctan(2*A[l, k] / (A[l, l]-A[k, k]))
        U_k = np.eye(n)
        U_k[l, l] = np.cos(phi)
        U_k[k, k] = np.cos(phi)
        U_k[l, k] = -np.sin(phi)
        U_k[k, l] = np.sin(phi)

        # print (U_k.T, A, U_k)
        A = U_k.T @ A @ U_k
        U @= U_k

        print(f"Iteration №{num}\nU_{num}=\n", U_k, f"\nA_{num}=\n", A)
        num += 1

        norm = matrix_norm(A)
        print(f"║A_{num}║ = {norm}, EPS = {EPS}")
        if norm < EPS:
            converged = True
    
    return np.diag(A).copy(), U


def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        A = np.matrix(data["A"], dtype=np.float32)
        EPS = data["EPS"]

    eigenvalues, eigenvectors = rotation_method(A, EPS)
    eigenvalues.sort()

    print("\nCalculated eignevalues using Rotation Method: ", eigenvalues)
    print("Calculated eignevectors using Rotation Method:\n", eigenvectors)

    numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(A)
    numpy_eigenvalues.sort()
    print("\nCalculated eignevalues using NumPy function:", numpy_eigenvalues)
    print("Calculated eignevectors using NumPy function:\n", numpy_eigenvectors)


cur_dir = Path(__file__).parent
solve(cur_dir / "input.json" )
