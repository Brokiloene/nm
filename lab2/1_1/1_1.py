from typing import Callable

import numpy as np


np.seterr(divide='ignore', invalid='ignore')

def f(x: float) -> float:
    return np.tan(x) - 5*x*x + 1

def df(x: float) -> float:
    return 1/(np.cos(x)**2) - 10*x

def phi(x: float) -> float:
    return (1 + np.tan(x)) / (5*x)

def dphi(x: float) -> float:
    return (2*x - 2*np.cos(x)**2 - np.sin(2*x)) / (10*x**2 * np.cos(x)**2)

def calc_q(a: float, b: float, dphi: Callable[[float], float]) -> float:
    sp = np.linspace(a,b,20)
    q = max([dphi(x) for x in sp])
    return q

def relax(x_p: float, x_prev: float):
    w = 0.5
    return w*x_p + (1-w)*x_prev

def simple_iterations(a:float, b:float, EPS:float) -> float|None:
    if f(a)*f(b) >= 0:
        print(f"Simple iterations method error: f({a})*f({b}) >= 0")
        return None
    
    q = calc_q(a,b,dphi)
    print(f"EPS = {EPS}, q = {q}")
    x_i = (a + b) / 2

    i = 1
    converged = False
    while not converged:
        x_prev = x_i
        # x_i = relax(phi(x_i), x_prev)
        x_i = phi(x_i)

        EPS_i = q/(1-q) * np.abs(x_i - x_prev)

        print(f"Iteration №{i}: x_i={x_i}, EPS_i={EPS_i}")
        i += 1
        if EPS_i <= EPS:
            converged = True       

    return x_i

def newton_method(a:float, b:float, EPS:float) -> float|None:
    if f(a)*f(b) >= 0:
        print(f"Newton method error: f({a})*f({b}) >= 0")
        return None

    x_i = (a + b) / 2

    i = 1
    converged = False
    while not converged:
        x_prev = x_i

        delta_x = -f(x_prev) / df(x_prev)
        x_i = x_prev + delta_x
        
        EPS_i = abs(x_i - x_prev)

        print(f"Iteration №{i}: x_i={x_i}, EPS_i={EPS_i}")
        i += 1
        if EPS_i < EPS:
            converged = True
    return x_i


def solve():
    EPS = 0.0001

    print("\nSimple iterations")
    x = simple_iterations(0.25, 1, EPS)
    print(f"Answer x={x}, f(x) = {f(x)}")

    print("\nNewton method")
    x = newton_method(0.25, 1, EPS)
    print(f"Answer x={x}, f(x) = {f(x)}")

solve()