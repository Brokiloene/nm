import json

import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def matrix_norm(A: np.matrix):
    A = np.fabs(A)
    max_col = np.max(np.sum(A, axis=0,))
    max_row = np.max(np.sum(A, axis=1))
    return max(max_col, max_row)


def vector_norm(b: np.array):
    b = np.fabs(b)
    return np.max(b)


def simple_iteration(A: np.matrix, b: np.array, EPS) -> np.array:
    n = A.shape[0]

    Alpha = np.zeros(A.shape) 
    Beta = np.zeros(n)

    # Jacobi method
    for i in range(n):
        Beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i == j:
                continue
            Alpha[i, j] = -A[i, j] / A[i, i]

    norm_alpha = matrix_norm(Alpha)
    norm_beta = vector_norm(Beta)

    print("Alpha:\n", Alpha)
    print("Beta:\n", Beta)
    print(f"║Alpha║ = {norm_alpha}, ║Beta║ = {norm_beta}")

    if norm_alpha < 1:
        suf_cond = True
        print("\n║Alpha║ < 1 => Sufficient condition is fulfilled")

        estimated_steps = (np.log10(EPS) - np.log10(norm_beta) + np.log10(1.0-norm_alpha)) / np.log10(norm_alpha)
        estimated_steps = int(np.floor(estimated_steps))
        print(f"Estimated steps: {estimated_steps}\n")

    else:
        suf_cond = False
        print("\n║Alpha║ >= 1 => Sufficient condition is not fulfilled\n")


    x: np.array = Beta.copy()

    i = 1
    while True:
        x_i: np.array = Beta + Alpha.dot(x)
        if suf_cond:
            eps_i = (norm_alpha / (1.0-norm_alpha)) * vector_norm(x_i - x)
        else:
            eps_i = vector_norm(x_i - x)
        
        x = x_i.copy()
        
        print(f"Iteration №{i}, EPS = {EPS}, x_i=", x, f" eps_i = {eps_i}")

        i += 1

        if eps_i < EPS:
            break
            
    return x


def seidel_method(A: np.matrix, b: np.array, EPS) -> np.array:
    n = A.shape[0]

    Alpha = np.zeros(A.shape) 
    Beta = np.zeros(n)

    # Jacobi method
    for i in range(n):
        Beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i == j:
                continue
            Alpha[i, j] = -A[i, j] / A[i, i]
    
    norm_alpha = matrix_norm(Alpha)
    norm_beta = vector_norm(Beta)

    print("Alpha:\n", Alpha)
    print("Beta:\n", Beta)
    print(f"║Alpha║ = {norm_alpha}, ║Beta║ = {norm_beta}")

    if norm_alpha < 1:
        suf_cond = True
        print("\n║Alpha║ < 1 => Sufficient condition is fulfilled")
    else:
        suf_cond = False
        print("\n║Alpha║ >= 1 => Sufficient condition is not fulfilled\n")

    x: np.array = Beta.copy()

    Alpha_1 = np.zeros(Alpha.shape)
    Alpha_2 = np.zeros(Alpha.shape)

    for i in range(n):
        for j in range(n):
            if i < j:
                Alpha_1[i, j] = Alpha[i, j]
            else:
                Alpha_2[i, j] = Alpha[i, j]
    # print("Alpha_1:\n", Alpha_1)
    # print("Alpha_2:\n", Alpha_2)

    norm_alpha2 = matrix_norm(Alpha_2)

    num = 1
    while True:
        x_i: np.array = Beta.copy()
        # x_i = Beta + Alpha_1.dot(x_i) + Alpha_2.dot(x)
        for i in range(n):
            for j in range(0, i):
                # print(f"Alpha_2[i, j]*x_i[j]: {Alpha_2[i, j]} * {x_i[j]}")
                x_i[i] += Alpha_2[i, j]*x_i[j]

            for j in range(i+1, n):
                # print(f"Alpha_1[i, j]*x[j]: {Alpha_1[i, j]} * {x[j]}")
                x_i[i] += Alpha_1[i, j]*x[j]
            

        if suf_cond:
            eps_i = (norm_alpha2 / (1.0-norm_alpha)) * vector_norm(x_i - x)
        else:
            eps_i = vector_norm(x_i - x)
        
        x = x_i.copy()

        if eps_i < EPS:
            break

        print(f"Iteration №{num}, EPS = {EPS}, x_i=", x, f" eps_i = {eps_i}")
        num += 1
    
    return x

def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        A = np.matrix(data["A"], dtype=np.float32)
        b = np.array(data["b"], dtype=np.float32)
        EPS = data["EPS"]
    
    print(f"\n{'Simple iteration method':=^30}\n")
    x = simple_iteration(A, b, EPS)
    print("\nSolution x:\n", x)

    n = A.shape[0]
    check_x = np.zeros(x.shape)

    for i in range(n):
        for j in range(n):
            check_x[i] += A[i, j]*x[j]
    
    print("Solution check: b\n", b, "\nwith x:\n", check_x)

    print(f"\n{'Seidel method':=^30}\n")
    x = seidel_method(A, b, EPS)

    print("\nSolution x:\n", x)


    
solve("./lab1/1_3/input3.json")
