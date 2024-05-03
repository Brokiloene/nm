import numpy as np
import matplotlib.pyplot as plt

from typing import List


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)


class LagrangePolynomial:
    def __init__(self, numenators: np.array, denumenators: np.array, foo_vals: np.array, repr_str: str) -> None:
        self.nums = numenators
        self.denums = denumenators
        self.yi = foo_vals
        self.repr_str = repr_str
    
    def __str__(self) -> str:
        return self.repr_str
    
    def __call__(self, x: float) -> float:
        ans = 0
        for i in range(len(self.nums)):
            cur_x = 1
            for s in self.nums[i]:
                cur_x *= (x + s)
            cur_x = cur_x * self.yi[i] / self.denums[i]
            ans += cur_x
        return ans


def f(x: float) -> float:
    return np.arccos(x) + x

def calc_li(xi: np.array, yi: np.array, ind: int) -> tuple[List[float], float, str]:
    n = len(xi)

    numstr = f"{yi[ind]:.2f}*"
    numenator = np.zeros(n-1)
    denumenator:float = 1

    cur_i = 0
    for i, x in enumerate(xi):
        if i == ind:
            continue

        numenator[cur_i] = -x
        cur_i += 1

        numstr += f"(x-{x:.2f})"
        denumenator *= (xi[ind]-x)

    ans = numstr + '/' + f"{denumenator:.2f}"
    return numenator, denumenator, ans
    

def lagrange_interpolation(xi: np.array, yi: np.array) -> LagrangePolynomial:
    n = len(xi)
    ans = "L(x) = "

    numenators = np.zeros(shape=(n, n-1))
    denumenators = np.zeros(shape=(n))

    for i in range(len(xi)):
        numenator, denumenator, curans = calc_li(xi, yi, i)

        numenators[i] = numenator
        denumenators[i] = denumenator

        if i < len(xi)-1:
            ans += curans + " + "
        else:
            ans += curans
    return LagrangePolynomial(numenators, denumenators, yi, ans)


class NewtonPolynomial:
    def __init__(self, coeffs: np.array, xi: np.array, repr_str: str) -> None:
        self.coeffs = coeffs.copy()
        self.xi = xi
        self.repr_str = repr_str
        
    def __str__(self) -> str:
        return self.repr_str
    
    def __call__(self, x: float) -> float:
        ans = 0
        n = len(self.coeffs)
        for i in range(n):
            cur = self.coeffs[0][i]
            for j in range(i):
                cur *= (x - self.xi[j])
            ans += cur

        return ans


def calc_divdiffs(xi: np.array, yi: np.array) -> np.matrix:
    n = len(xi)
    coeffs = np.zeros([n, n])
    coeffs[:,0] = yi

    for j in range(1,n):
        for i in range(n-j):
            coeffs[i, j] = (coeffs[i+1, j-1] - coeffs[i, j-1]) / (xi[i+j] -xi[i])
    
    return coeffs

def newtone_interpolation(xi: np.array, yi: np.array):
    n = len(xi)
    coeffs = calc_divdiffs(xi, yi)

    ans = "N(x) = "
    for i in range(n):
        ans += f"{coeffs[0][i]:.2f}"

        for j in range(i):
            ans += f"(x-{xi[j]:.2f})"
        
        if i < n-1:
            ans += " + "

    return NewtonPolynomial(coeffs, xi, ans)


def solve():
    xi = np.array([-0.4, -0.1, 0.2, 0.5])
    xi = np.array([-0.4, 0, 0.2, 0.5])
    yi = np.array([f(x) for x in xi])

    # xi = np.array([0,1,2,3])
    # yi = np.array([0,0.5,0.86,1.0])
    xS = 0.1
    
    lp = lagrange_interpolation(xi, yi)
    for x in xi:
        print(f"f({x}) = {f(x)}")
    print(f"f(x*) = {f(xS)}")

    print("\nLagrange interpolation")
    print(lp)
    for x in xi:
        print(f"L({x}) = {lp(x)}")
    print(f"L(x*) = {lp(xS)}")
    print(f"Error: {abs(lp(xS) - f(xS))}")


    nwp = newtone_interpolation(xi, yi)
    print("\nNewton interpolation")
    print(nwp)
    for x in xi:
        print(f"N({x}) = {nwp(x)}")
    print(f"N(x*) = {nwp(xS)}")
    print(f"Error: {abs(nwp(xS) - f(xS))}")


    xx = np.linspace(-1, 1, 10)
    fy = np.array([f(x) for x in xx])
    ly = np.array([lp(x) for x in xx])
    ny = np.array([nwp(x) for x in xx])

    plt.plot(xx, fy, color="black", label="arccos(x)+x")
    plt.plot(xx, ly, color="red", label="Lagrange")
    plt.plot(xx, ny, color="blue", label="Newton")
    plt.legend()
    plt.show()


solve()

