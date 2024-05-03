import numpy as np


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
    
    # print("P:\n", p)
    # print("Q:\n", q)

    return x