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
plt.plot(x_vals, y_vals, label=r'$f(x)$', color='red')

# Przerywane linie bez dodawania do legendy
plt.axvline(a, color='gray', linestyle='--')
plt.axvline(b, color='gray', linestyle='--')

# Zwiększone czcionki
font_size = 14
plt.xlabel(r'wartość metryki', fontsize=font_size)
plt.ylabel(r'dopasowanie', fontsize=font_size)

plt.xticks([a - delta, a, b, b + delta],
           [r'$a - (b - a)$', r'$a$', r'$b$', r'$b + (b - a)$'],
           fontsize=font_size)
plt.yticks([0, 0.5, 1], fontsize=font_size)

plt.grid(True)
plt.legend(fontsize=font_size)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.show()
