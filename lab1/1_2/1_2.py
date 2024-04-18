import json
from pathlib import Path

import numpy as np


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def tdma(a: np.array, b: np.array, c: np.array, d: np.array) -> np.array:
    n = len(b)

    p = np.zeros(n)
    q = np.zeros(n)

    for i in range(n):
        if i == 0:
            p[i] = -c[i] / b[i]
            q[i] = d[i] / b[i]
        else:
            if i != n-1:
                # print(f"i={i}, c[i]={c[i]}, a[i]={a[i]}, p[i-1]={p[i-1]}")
                p[i] = -c[i] / (b[i] + a[i]*p[i-1])

            q[i] = (d[i] - a[i]*q[i-1]) / (b[i] + a[i]*p[i-1])
    
    x = np.zeros(n)
    x[n-1] = q[n-1]

    for i in range(n-2, -1, -1):
        x[i] = p[i]*x[i+1] + q[i]
    
    print("P:\n", p)
    print("Q:\n", q)

    return x

def solve(inputfile):
    with open(inputfile, "r") as f:
        data = json.load(f)

        a = np.array(data["a"], dtype=np.float32)
        b = np.array(data["b"], dtype=np.float32)
        c = np.array(data["c"], dtype=np.float32)
        d = np.array(data["d"], dtype=np.float32)
    
    x = tdma(a,b,c,d)
    print("\nSolution x:\n", x)

    n = len(b)
    check_x = np.zeros(n)

    for i in range(n):
        if i == 0:
            check_x[i] = b[i]*x[i] + c[i]*x[i+1]
        elif i == n-1:
            check_x[i] = a[i]*x[i-1] + b[i]*x[i]
        else:
            check_x[i] = a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]
    
    print("Solution check: d\n", d, "\nwith x:\n", check_x)

cur_dir = Path(__file__).parent
solve(cur_dir / "input.json" )
