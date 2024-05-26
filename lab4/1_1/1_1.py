import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, floatmode="fixed", suppress=True)

a = 2
b = 3
h = 0.1
y0 = np.sqrt(2)*2
dy0 = np.sqrt(2)*1.5


def f(x,y,z):
    return (0.75*y-0.5*z) / (x * (x - 1))

def f_true(x):
    return np.abs(x)**1.5

def euler_explicit(f, y0, z0, a, b, h, print_vals=True):
    xs = np.arange(a,b+h, h)
    ys = [y0]
    y_i = y0
    z_i = z0

    for i in range(len(xs)-1):
        z_i += h*f(xs[i],ys[i],z_i)
        y_i = ys[i] + h*z_i

        if print_vals:
            print(f"x{i} = {xs[i] :<20} y{i} = {y_i :<20}")
        ys.append(y_i)
    return xs, ys

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
        

def adams_method(f, y, z, a, b, h, print_vals=True):
    # xs = x[:4]
    xs = np.arange(a, b+h, h)
    ys = y[:4]
    zs = z[:4]
    for i in range(3, len(xs) - 1):
        z_i = zs[i] + h * (55 * f(xs[i], ys[i], zs[i]) -
                          59 * f(xs[i - 1], ys[i - 1], zs[i - 1]) +
                          37 * f(xs[i - 2], ys[i - 2], zs[i - 2]) -
                           9 * f(xs[i - 3], ys[i - 3], zs[i - 3])) / 24
        zs.append(z_i)
        y_i = ys[i] + h * (55 * zs[i] -
                          59 * zs[i - 1] +
                          37 * zs[i - 2] -
                           9 * zs[i - 3]) / 24
        ys.append(y_i)
        # xs = np.append(xs, xs[i] + h)
        # xs.append(xs[i] + h)
        if print_vals:
            print(f"x{i} = {xs[i] :<20} y{i} = {ys[i] :<20}")

    return xs, ys


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


x_true = np.arange(a, b+h, h)
y_true = [f_true(x) for x in x_true]


print("EULER METHOD\n")

xs, ys = euler_explicit(f, y0, dy0, a, b, h)
_, y2s = euler_explicit(f, y0, dy0, a, b, h/2, print_vals=False)

plt.subplot(131)
plt.scatter(xs, ys, color="red")
plt.plot(xs, ys, color="red", alpha=0.5, label="euler")

plt.scatter(x_true, y_true, color="green")
plt.plot(x_true, y_true, color="green", alpha=0.5, label="true")
plt.legend()

print("Runge-Romberg error: ", runge_romberg(ys, y2s))
print("Exact error: ", exact_error(ys, y_true))
print()


print("RUNGE-KUTTA METHOD\n")

xs, ys, zs = runge_kutta(f, y0, dy0, a, b, h)
_, y2s, _ = runge_kutta(f, y0, dy0, a, b, h/2)

plt.subplot(132)
plt.scatter(xs, ys, color="red")
plt.plot(xs, ys, color="red", alpha=0.5, label="runge-kutta")

plt.scatter(x_true, y_true, color="green")
plt.plot(x_true, y_true, color="green", alpha=0.5, label="true")
plt.legend()

print("Runge-Romberg error: ", runge_romberg(ys, y2s))
print("Exact error: ", exact_error(ys, y_true))
print()

print("ADAMS METHOD\n")

xs, ys = adams_method(f, ys, zs, a, b, h)
_, y2s = adams_method(f, ys, zs, a, b, h/2, print_vals=False)

plt.subplot(133)
plt.scatter(xs, ys, color="red")
plt.plot(xs, ys, color="red", alpha=0.5, label="adams")

plt.scatter(x_true, y_true, color="green")
plt.plot(x_true, y_true, color="green", alpha=0.5, label="true")
plt.legend()

print("Runge-Romberg error: ", runge_romberg(ys[4:], y2s[5:]))
print("Exact error: ", exact_error(ys, y_true))
print()

plt.show()
