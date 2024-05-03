import matplotlib.pyplot as plt
import numpy as np


def foo(x: float) -> float:
    return (1 + np.tan(x)) / (5*x)

xs = np.linspace(-1, 1, 20)
ys = np.array([foo(x) for x in xs])

plt.plot(xs, ys)

xs = np.linspace(-1, 1, 20)
ys = np.array([(lambda x: x)(x) for x in xs])
print(xs, ys)
plt.plot(xs,ys)

plt.show()

