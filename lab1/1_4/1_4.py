import json
from pathlib import Path

import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def matrix_norm(A: np.matrix) -> float:
    n = A.shape[0]
    res = np.sqrt( sum(A[i,j]**2 if i < j else 0 for i in range(n) for j in range(n)) )
    return res


def rotation_method(A: np.matrix, EPS) -> np.array:
    n = A.shape[0]

    converged = False
    num = 1
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
        U = np.eye(n)
        U[l, l] = np.cos(phi)
        U[k, k] = np.cos(phi)
        U[l, k] = -np.sin(phi)
        U[k, l] = np.sin(phi)

        # print (U.T, A, U)
        A = U.T @ A @ U

        print(f"Iteration №{num}\nU_{num}=\n", U, f"\nA_{num}=\n", A)
        num += 1

        norm = matrix_norm(A)
        print(f"║A_{num}║ = {norm}, EPS = {EPS}")
        if norm < EPS:
            converged = True
    
    return np.diag(A)


def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        A = np.matrix(data["A"], dtype=np.float32)
        EPS = data["EPS"]

    eigenvalues = np.sort(rotation_method(A, EPS))

    print("\nCalculated eignevalues using Rotation Method: ", eigenvalues)

    numpy_eigenvalues = np.sort(np.linalg.eig(A)[0])
    print("Calculated eignevalues using NumPy function:", numpy_eigenvalues)


cur_dir = Path(__file__).parent
solve(cur_dir / "input.json" )
