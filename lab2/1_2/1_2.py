from typing import Tuple

import numpy as np

from lu import solve_SLAE


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)
np.seterr(divide='ignore', invalid='ignore')

def f1(x1: float, x2: float):
    return x1**2 - 2*np.log10(x2) - 1
def f2(x1: float, x2: float):
    return x1**2 - 2*x1*x2 + 2

def d1f1(x1: float, x2: float):
    return 2*x1
def d2f1(x1: float, x2: float):
    return 2/(np.log(10)*x2)
def d1f2(x1: float, x2: float):
    return 2*x1 - 2*x2
def d2f2(x1: float, x2: float):
    return -2*x1


# def f1(x1: float, x2: float):
#     return 0.1 * x1**2 + x1 + 0.2*x2**2 - 0.3
# def f2(x1: float, x2: float):
#     return 0.2 * x1**2 + x2 - 0.1*x1*x2 - 0.7

# def d1f1(x1: float, x2: float):
#     return 0.2*x1+1
# def d2f1(x1: float, x2: float):
#     return 0.4*x2
# def d1f2(x1: float, x2: float):
#     return 0.4*x1 - 0.1*x2
# def d2f2(x1: float, x2: float):
#     return 1 - 0.1*x1

def relax(x_p: float, x_prev: float):
    w = 0.5
    return w*x_p + (1-w)*x_prev

def newton(a1: float, b1: float, a2: float, b2: float, EPS: float, use_LU=True) -> Tuple[float, float]:
    x1_k = (a1 + b1) / 2
    x2_k = (a2 + b2) / 2

    shape = (2,2)
    J = [[d1f1, d2f1],
         [d1f2, d2f2]]
    
    if not use_LU:
        A1 = [[f1, d2f1],
            [f2, d2f2]]
        A2 = [[d1f1, f1],
            [d1f2, f2]]
    
    k = 1
    converged = False
    while not converged:
        J_k = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                J_k[i,j] = J[i][j](x1_k,x2_k)
        
        x1_prev = x1_k
        x2_prev = x2_k

        if use_LU:
            A = J_k.copy()
            b = np.array([-f1(x1_k, x2_k), -f2(x1_k, x2_k)])

            delta_x = solve_SLAE(A, b)
            x1_k += delta_x[0]
            x2_k += delta_x[1]

        if not use_LU:
            A1_k = np.zeros(shape)
            A2_k = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    A1_k[i,j] = A1[i][j](x1_k,x2_k)
                    A2_k[i,j] = A2[i][j](x1_k,x2_k)
            
            x1_k = x1_k - np.linalg.det(A1_k) / np.linalg.det(J_k)
            x2_k = x2_k - np.linalg.det(A2_k) / np.linalg.det(J_k)

        EPS_k = max(abs(x1_k - x1_prev), abs(x2_k - x2_prev))
        print(f"Iteration №{k}: x1_{k}={x1_k}, x2_{k}={x2_k} EPS_{k}={EPS_k}")

        k += 1
        if EPS_k <= EPS:
            converged = True
    return (x1_k, x2_k)

def phi1(x1: float, x2: float):
    return np.sqrt(2*np.log10(x2) + 1)
def phi2(x1: float, x2: float):
    return (x1**2 + 2) / (2 * x1)

def d1p1(x1: float, x2: float):
    return 0
def d2p1(x1: float, x2: float):
    return 1/(np.log(10) * x1 * np.sqrt(2*np.log10(x1) + 1))
def d1p2(x1: float, x2: float):
    return 0.5 - 1/(x1**2)
def d2p2(x1: float, x2: float):
    return 0

def calc_q(a1: float, b1: float, a2: float, b2: float):
    x1s = np.linspace(a1,b1,20)
    x2s = np.linspace(a2,b2,20)
    
    q1 = max([abs(d1p1(x1, x2)) + abs(d2p1(x1, x2)) for x1 in x1s for x2 in x2s])
    q2 = max([abs(d1p2(x1, x2)) + abs(d2p2(x1, x2)) for x1 in x1s for x2 in x2s])

    return max(q1, q2)


def simple_iterations(a1: float, b1: float, a2: float, b2: float, EPS: float) -> Tuple[float, float]:

    x1_k = (a1 + b1) / 2
    x2_k = (a2 + b2) / 2
    q = calc_q(a1, b1, a2, b2)

    k = 1
    converged = False
    while not converged:
        x1_prev = x1_k
        x2_prev = x2_k

        x1_k = phi1(x1_prev, x2_prev)        
        x2_k = phi2(x1_prev, x2_prev)
        norm = max([abs(x1_k - x1_prev), abs(x2_k - x2_prev)])

        EPS_k = norm*q/(1-q)

        print(f"Iteration №{k}: x1_{k} = {x1_k}, x2_{k} = {x2_k} EPS_{k}={EPS_k}")
        k += 1
        if EPS_k < EPS:
            converged = True
    return x1_k, x2_k


def solve():
    print("===Newton Method===")
    x1, x2 = newton(1, 2, 1, 2, 0.0001, use_LU=False)
    print(f"x1 = {x1}, x2 = {x2}")
    print(f"f1(x1,x2) = {f1(x1,x2)}, f2(x1,x2) = {f2(x1,x2)}")

    print("\n===Simple Iterations===")
    x1, x2 = simple_iterations(1, 2, 1, 2, 0.0001)
    print(f"x1 = {x1}, x2 = {x2}")
    print(f"f1(x1,x2) = {f1(x1,x2)}, f2(x1,x2) = {f2(x1,x2)}")
    

solve()