import numpy as np
import matplotlib.pyplot as plt

def dopasowanie_symboliczne(x, a, b):
    if x < a:
        return max(0, 1 - (a - x) / (b - a))
    elif x > b:
        return max(0, 1 - (x - b) / (b - a))
    else:
        return 1

a = 3
b = 7
delta = b - a

x_min = a - delta - 1
x_max = b + delta + 1
x_vals = np.linspace(x_min, x_max, 500)
y_vals = [dopasowanie_symboliczne(x, a, b) for x in x_vals]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label=r'$f(x)$', color='blue')
plt.axvline(a, color='gray', linestyle='--', label=r'$x = a$')
plt.axvline(b, color='gray', linestyle='--', label=r'$x = b$')
plt.xlabel(r'$x$ (wartość metryki)')
plt.ylabel(r'$f(x)$ (dopasowanie)')
plt.title(r'Wykres funkcji dopasowania względem zakres idealnego $[a, b]$')
plt.xticks([a - delta, a, b, b + delta],
           [r'$a - (b - a)$', r'$a$', r'$b$', r'$b + (b - a)$'])
plt.yticks([0, 0.5, 1])
plt.grid(True)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show()
