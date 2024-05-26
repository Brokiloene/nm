from tools import runge_kutta, runge_romberg, exact_error, tdma

import numpy as np
import matplotlib.pyplot as plt


np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

def f_true(x):
    return 2*x + 1 + np.e ** (2*x)

def f(x, y, z):
    return (-2*y + (2*x+1)*z) / x


# a * y' + b * y = c
# cond = [a, b, c]

# cond1 = [1,0,4]
# 1 * y' + 0 * y = 4
# cond2 = [1,-2,-4]
# 1 * y' - 2 * y = -4

def shooting_method(f, cond1, cond2, a, b, h, EPS):
    def phi(n):
        if abs(cond1[0]) > EPS:
            y0 = n
            z0 = (cond1[2] - n*cond1[1]) / cond1[0]
        else:
            y0 = cond1[2] / cond1[1]
            z0 = n
        _, ys, zs = runge_kutta(f, y0, z0, a, b, h, print_vals=False)

        return zs[-1]*cond2[0] + cond2[1]*ys[-1] - cond2[2] 
        # return zs[-1] - 2*ys[-1] + 4 

    n1, n2 = 1.0, 0.8

    converged = False
    while not converged:
        
        n = n2 - (n2 - n1) / (phi(n2) - phi(n1)) * phi(n2)
        n1 = n2
        n2 = n
        if np.abs(phi(n)) < EPS:
            converged = True
    
    if abs(cond1[0]) > EPS:
        y0 = n2
        z0 = (cond1[2] - n2*cond1[1]) / cond1[0]
    else:
        y0 = cond1[2] / cond1[1]
        z0 = n2

    xi, yi, _ = runge_kutta(f, y0, z0, a, b, h, print_vals=False)
    return xi, yi



def p(x):
    return -(2*x + 1) / x

def q(x):
    return 2 / x

def foo(x):
    return 0

# tdma
# [b, c, 0, 0] = [d]
# [a, b, c, 0] = [d]
# [0, a, b, c] = [d]
# [0, 0, a ,b] = [d]

# a * y' + b * y = c
# cond = [a, b, c]

def finite_difference(p, q, f, l, r, h, cond1, cond2):
    xs = np.arange(l, r+h, h)
    # l += 0.0001
    n = len(xs)
    
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    a[0] = 0
    b[0] = cond1[1]*h - cond1[0]
    c[0] = cond1[0]
    d[0] = cond1[2]*h

    a[-1] = -cond2[0]
    b[-1] = cond2[0] + cond2[1]*h
    c[-1] = 0
    d[-1] = cond2[2]*h

    for i in range(1, n-1):
        a[i] = 1 - p(xs[i]) * h / 2
        b[i] = -2 + h**2 * q(xs[i])
        c[i] = 1 + p(xs[i]) * h / 2
        d[i] = h**2 * f(xs[i])
    
    ys = tdma(a, b, c, d)
    return xs, ys

def print_table(xs, ys, ys_true):
    for i in range(len(xs)):
        print(f"x{i} = {xs[i] :<20} y{i} = {ys[i] :<20} y_true{i} = {y_true[i] :<20}")


a = 0.001
b = 1
h = 0.1
EPS = 0.001

cond1 = [1,0,4]
cond2 = [1,-2,-4]

x_true = np.arange(a, b+h, h)
y_true = [f_true(x) for x in x_true]


print("\nSHOOTING METHOD")
xs, ys = shooting_method(f, cond1, cond2, a, b, h, EPS)
_, y2s = shooting_method(f, cond1, cond2, a, b, h/2, EPS)

print_table(xs, ys, y_true)
print("Runge-Romberg error: ", runge_romberg(ys, y2s, p=1))
print("Exact error: ", exact_error(ys, y_true))
print()

plt.subplot(121)
plt.plot(xs, ys, 's-', color="red", alpha=0.5, label="shooting")
plt.plot(x_true, y_true, 'o-', color="blue", alpha=0.5, label="true")
plt.legend()


print("\nFINITE DIFFERENCE METHOD")
xs, ys = finite_difference(p, q, foo, a, b, h, cond1, cond2)
_, y2s = finite_difference(p, q, foo, a, b, h/2, cond1, cond2)

print_table(xs, ys, y_true)
print("Runge-Romberg error: ", runge_romberg(ys, y2s, p=1))
print("Exact error: ", exact_error(ys, y_true))
print()

plt.subplot(122)
plt.plot(xs, ys, 'o-', color="red", alpha=0.5, label="finite diff")
plt.plot(x_true, y_true, 'o-', color="blue", alpha=0.5, label="true")
plt.legend()

plt.show()
