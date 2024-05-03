import json
from pathlib import Path

import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def calc_df(xi: np.array, yi: np.array, xS: float) -> float:
    def diff(i1: int, i2: int):
        nonlocal xi
        nonlocal yi
        return (yi[i1]-yi[i2])/(xi[i1]-xi[i2]) 

    for ii in range(len(xi)-1):
        if xi[ii] <= xS < xi[ii+1]:
            i = ii
            break
    
    return diff(i+1,i) + \
           (diff(i+2, i+1) - diff(i+1,i))/(xi[i+2]-xi[i])*(2*xS-xi[i]-xi[i+1])

def calc_d2f(xi: np.array, yi: np.array, xS: float) -> float:
    def diff(i1: int, i2: int):
        nonlocal xi
        nonlocal yi
        return (yi[i1]-yi[i2])/(xi[i1]-xi[i2]) 

    for ii in range(len(xi)-1):
        if xi[ii] <= xS < xi[ii+1]:
            i = ii
            break
    
    return 2*(diff(i+2,i+1)-diff(i+1,i))/(xi[i+2]-xi[i])

def df(x, y, x_):
    assert len(x) == len(y)
    for interval in range(len(x)):
        if x[interval] <= x_ < x[interval+1]:
            i = interval
            break

    a1 = (y[i+1] - y[i]) / (x[i+1] - x[i])
    a2 = ((y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - a1) / (x[i+2] - x[i]) * (2*x_ - x[i] - x[i+1])
    return a1 + a2


def d2f(x, y, x_):
    assert len(x) == len(y)
    for interval in range(len(x)):
        if x[interval] <= x_ < x[interval+1]:
            i = interval
            break

    num = (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i])
    return 2 * num / (x[i+2] - x[i])

def solve(inputfile: str):
    with open(inputfile, "r") as f:
        data = json.load(f)

        xi = np.array(data["x"], dtype=np.float32)
        yi = np.array(data["y"], dtype=np.float32)
        xS = data["x*"]
    
    print(f"f'(xS) = {calc_df(xi,yi,xS)}")
    print(f"f''(xS) = {calc_d2f(xi,yi,xS)}")

cur_dir = Path(__file__).parent
solve(cur_dir / "input.json" )
