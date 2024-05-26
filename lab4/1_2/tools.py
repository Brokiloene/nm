import numpy as np

def runge_kutta(f, y0, z0, a, b, h, print_vals=True):
    xs = np.arange(a, b+h, h)
    ys = [y0]
    zs = [z0]

    for i in range(len(xs)-1):
        K1 = h * zs[i]
        L1 = h * f(xs[i], ys[i], zs[i])

        K2 = h * (zs[i] + 0.5 * L1)
        L2 = h * f(xs[i] + 0.5 * h, 
                   ys[i] + 0.5 * K1,
                   zs[i] + 0.5 * L1)

        K3 = h * (zs[i] + 0.5 * L2)
        L3 = h * f(xs[i] + 0.5 * h, 
                   ys[i] + 0.5 * K2,
                   zs[i] + 0.5 * L2)

        K4 = h * (zs[i] + L3)
        L4 = h * f(xs[i] + h, 
                   ys[i] + K3,
                   zs[i] + L3)
        
        delta_y = (K1 + 2*K2 + 2*K3 + K4) / 6
        delta_z = (L1 + 2*L2 + 2*L3 + L4) / 6
        ys.append(ys[i] + delta_y)
        zs.append(zs[i] + delta_z)

        if print_vals:
            print(f"x{i} = {xs[i] :<20} y{i} = {ys[i] :<20}")
    
    return xs, ys, zs

def runge_romberg(y1, y2, p=2):
    error = 0
    for i in range(len(y1)):
        y_true = (y1[i] + (y1[i] - y2[i*2]) / (2**p - 1))
        error += np.abs(y1[i] - y_true)
    return error / len(y1)


def exact_error(y1, y2):
    error = 0
    for i in range(len(y1)):
        error += np.abs(y1[i] - y2[i])
    return error / len(y1)

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
                p[i] = -c[i] / (b[i] + a[i]*p[i-1])

            q[i] = (d[i] - a[i]*q[i-1]) / (b[i] + a[i]*p[i-1])
    
    x = np.zeros(n)
    x[n-1] = q[n-1]

    for i in range(n-2, -1, -1):
        x[i] = p[i]*x[i+1] + q[i]

    return x