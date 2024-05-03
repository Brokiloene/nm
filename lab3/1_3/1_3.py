import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lu import solve_SLAE


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)


class Polynomial:
    def __init__(self, a: np.array) -> None:
        self.a = a
        self.power = len(a)
    
    def __call__(self, x: float):
        ans = 0
        for i in range(self.power):
            ans += (self.a[i] * x**i)
        return ans

def mls(xi: np.array, yi: np.array, power: int) -> Polynomial:
    Phi = np.matrix([[x**j for j in range(power+1)] for x in xi])
    G = Phi.T @ Phi
    b = Phi.T @ yi
    b = np.squeeze(np.asarray(b))
    a = solve_SLAE(G, b)
    return Polynomial(a)

def least_squares(p: Polynomial, xi: np.array, yi: np.array) -> float:
    return sum([abs(p(xi[i])-yi[i]) for i in range(len(xi))])

def solve(inputfile: str):
    with open(inputfile, "r") as f:
        data = json.load(f)

        xi = np.array(data["x"], dtype=np.float32)
        yi = np.array(data["y"], dtype=np.float32)

    p1 = mls(xi, yi, 1)
    print(f"1st LS error: {least_squares(p1, xi, yi)}")
    p2 = mls(xi, yi, 2)
    print(f"2nd LS error: {least_squares(p2, xi, yi)}")

    plt.scatter(xi, yi, color="black")

    xx = np.linspace(xi[0], xi[-1], 20)
    plt.plot([x for x in xx], [p1(x) for x in xx], color="red", label="1st degree")
    plt.plot([x for x in xx], [p2(x) for x in xx], color="blue", label="2nd degree")
    
    # p3 = mls(xi, yi, 3)
    # print(f"2nd LS error: {least_squares(p3, xi, yi)}")
    # plt.plot([x for x in xx], [p3(x) for x in xx], color="orange", label="3rd degree")

    plt.legend()
    plt.show()



cur_dir = Path(__file__).parent
solve(cur_dir / "input.json" )
