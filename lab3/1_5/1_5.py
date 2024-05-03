import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)


def f(x: float):
    return x / (4+3*x)
# def f(x):
#     return x /(3*x+4)**2

def rectangle_method(xi: np.array) -> float:
    S = 0
    for i in range(len(xi)-1):
        xm = (xi[i] + xi[i+1]) / 2
        S += f(xm) * (xi[i+1] - xi[i])
    return S


def trapezoid_method(yi: np.array, h: float):
    S = h * ((yi[0] + yi[-1])/2 + sum(yi[i] for i in range(1,len(yi)-1)))
    return S

def simpson_method(yi: np.array, h: float):
    S = h/3 * (yi[0] + yi[-1] + sum(4*yi[i] if i%2==1 else 2*yi[i] for i in range(1,len(yi)-1)))
    return S

def runge_roberg(s1: float, s2: float, h1: float, h2: float, p=2):
    return s1 + (s1 - s2) / ((h2/h1)**p - 1)

def solve():
    x0 = 1
    xk = 5
    h1 = 1.0
    h2 = 0.5

    # x0 = -1
    # xk = 1
    # h1 = 0.5
    # h2 = 0.25

    xi1 = np.arange(x0, xk+h1, h1)
    yi1 = np.array([f(x) for x in xi1])

    xi2 = np.arange(x0, xk+h2, h2)
    yi2 = np.array([f(x) for x in xi2])

    S_real = 0.889543

    S_rec1 = rectangle_method(xi1)
    S_rec2 = rectangle_method(xi2)
    runge = runge_roberg(S_rec1, S_rec2, h1, h2)
    print("===Rectangle method===")
    print(f"h = {h1}, S =", S_rec1)
    print(f"h = {h2}, S =",S_rec2)
    print(f"Runge =", runge)
    print(f"Real =", S_real)
    print(f"Error = {runge - S_real}\n")

    S_tr1 = trapezoid_method(yi1, h1)
    S_tr2 = trapezoid_method(yi2, h2)
    runge = runge_roberg(S_tr1, S_tr2, h1, h2)
    print("===Trapezoid method===")
    print(f"h = {h1}, S =", S_tr1)
    print(f"h = {h2}, S =",S_tr2)
    print(f"Runge =", runge)
    print(f"Real =", S_real)
    print(f"Error = {runge - S_real}\n")



    S_sim1 = simpson_method(yi1, h1)
    S_sim2 = simpson_method(yi2, h2)
    runge = runge_roberg(S_sim1, S_sim2, h1, h2)
    print("===Simpson method===")
    print(f"h = {h1}, S =", S_sim1)
    print(f"h = {h2}, S =",S_sim2)
    print(f"Runge =", runge)
    print(f"Real =", S_real)
    print(f"Error = {runge - S_real}\n")




    # print(rectangle_method(xi))
    # print(runge_roberg())
    # print(trapezoid_method(yi, h2))
    # print(simpson_method(yi, h2))

solve()