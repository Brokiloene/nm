import json
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from tdma import tdma


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)


class Spline:
    def __init__(self, a: np.array, b: np.array, c: np.array, d: np.array, xi: np.array) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.xi = xi

    def __call__(self, x, x_o):
        def calc(x: float, ind: int):
            return self.a[ind] + \
                   self.b[ind]*x + \
                   self.c[ind]*x*x + \
                   self.d[ind]*x*x*x
        
        for i in range(len(self.xi)-1):
            if self.xi[i] <= x <= self.xi[i+1]:
                return calc(x-x_o, i)



def calc_coeffs(ci: np.array, h: np.array, yi: np.array) -> Tuple[np.array, 
                                                                  np.array, 
                                                                  np.array, 
                                                                  np.array]:
    
    n = len(yi)
    a = yi[:-1]
   
    b = np.array([(yi[i] - yi[i-1]) / h[i-1] - (1/3) * h[i-1] * (2*ci[i-1] + ci[i]) for i in range(1,n-1)])
    b = np.append(b, (yi[n-1]-yi[n-2])/h[n-2] - (2/3)*h[n-2]*ci[n-2])
   
    d = np.array([(ci[i] - ci[i-1]) / (3 * h[i-1]) for i in range(1,n-1)])
    d = np.append(d, -ci[n-2] / (3*h[n-2]))

    return (a,b,ci,d)


def spline_interpolation(xi: np.array, yi: np.array) -> Spline:
    n = len(xi)-1
    h = np.array([xi[i]-xi[i-1] for i in range(1,len(xi))])

    d1 = np.zeros(n-1)
    d2 = np.zeros(n-1)
    d3 = np.zeros(n-1)

    d2[0] = 2 * (h[0] + h[1])
    d3[0] = h[1]

    d1[n-2] = h[n-2]
    d2[n-2] = 2 * (h[n-2] + h[n-1])

    for i in range(1, n-2):
        d1[i] = h[i-1]
        d2[i] = 2 * (h[i-1] + h[i])
        d3[i] = h[i]

    b = np.array([3 * ((yi[i+1]-yi[i]) / h[i] - (yi[i] - yi[i-1]) / h[i-1]) for i in range(1, n)])

    ci = np.zeros(n)
    ci[1:] = tdma(d1,d2,d3,b)

    a,b,c,d = calc_coeffs(ci, h, yi)
    
    return Spline(a,b,c,d,xi) 


def solve(inputfile: str):
    with open(inputfile, "r") as f:
        data = json.load(f)

        xi = np.array(data["x"], dtype=np.float32)
        yi = np.array(data["y"], dtype=np.float32)
        xS = data["x*"]

    print("Spline interpolation")
    spline = spline_interpolation(xi, yi)

    for i in range(len(xi)-1):
        print(f"P_{i} = {spline.a[i]} + {spline.b[i]}*x + {spline.c[i]}*x^2 + {spline.d[i]}*x^3")

    print(f"S(x*) = {spline(xS,0)}")
    plt.scatter(xi, yi, color="black")

    for i in range(len(xi)-1):
        xx = np.linspace(xi[i]+0.001, xi[i+1], 10)
        plt.plot(xx, [spline(x, xi[i]) for x in xx])
    
    plt.show()


cur_dir = Path(__file__).parent
solve(cur_dir / "input.json" )
