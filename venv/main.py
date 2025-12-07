import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model(F, a1, a2, a3):
    return a1 / (1 + np.exp(-a2 * (F - a3)))

p_data = np.array([0.51, 0.58, 0.73, 0.85, 1.02, 1.18, 1.35, 1.48, 1.73, 2.01, 2.18, 2.51, 2.77, 2.91, 3.26, 3.48, 3.88, 4.15, 4.39, 4.75, 5.26, 5.55])
f_data = np.array([0, 754, 2513, 2882, 3173, 3349, 3492, 3610, 3739, 3824, 3854, 3903, 3950, 3954, 3975, 3984, 4002, 4009, 4016, 4021, 4030, 4033])
initial_guess = [4010, 5.0, 1.0]

params, covariance = curve_fit(
    model,
    p_data,
    f_data,
    p0=initial_guess,
    maxfev=10000
)

p_plot = np.linspace(min(p_data), max(p_data), 300)

f_initial = model(p_plot, *initial_guess)
f_fitted = model(p_plot, *params)

plt.figure()
plt.scatter(p_data, f_data, marker='o', label='Data')
plt.plot(p_plot, f_initial, linestyle='--', label='Initial guess')
plt.plot(p_plot, f_fitted, label='Fitted model')

plt.xlabel('p')
plt.ylabel('f')
plt.legend()
plt.grid(True)
plt.show()

a1, a2, a3 = params
print("Dopasowane parametry:")
print(f"a1 = {a1}")
print(f"a2 = {a2}")
print(f"a3 = {a3}")

f_fit = model(p_data, *params)

residuals = f_data - f_fit
SSR = np.sum(residuals**2)
SS_tot = np.sum((f_data - np.mean(f_data))**2)
R2 = 1 - SSR / SS_tot
print(R2)
