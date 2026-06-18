import numpy as np
from scipy import integrate, optimize


def f(x):
    return np.sin(x)


def fit_func(x, a, b, c):
    return a * x**2 + b * x + c


x = np.linspace(0, 10, 50)
y = 2.0 * x**2 + 3.0 * x + 1.0 + np.random.normal(scale=5.0, size=x.shape)

integral_value, _ = integrate.quad(f, 0, np.pi)
params, _ = optimize.curve_fit(fit_func, x, y)
root = optimize.brentq(f, 3, 4)

print(integral_value)
print(params)
print(root)
