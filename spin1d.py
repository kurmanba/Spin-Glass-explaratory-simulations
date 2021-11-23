import numpy as np
import matplotlib.pyplot as plt
import tqdm

np.random.seed(118)

N = 1_000_0

J = np.random.normal(0, 1, size=N - 1)

T_min = 1e-4
T_max = 50
T_steps = 1000
T = np.arange(T_steps) / T_steps * (T_max - T_min) + T_min
beta = 1 / T

F = np.zeros(T_steps)
U = np.zeros(T_steps)
C = np.zeros(T_steps)
S = np.zeros(T_steps)

for i in range(T_steps):
    F[i] = - T[i] * np.log(2) - T[i] * (np.log(2 * np.cosh(beta[i] * J))).sum()
    U[i] = - (J * np.tanh(J * beta[i])).sum()
    C[i] = ((beta[i] * J) ** 2 * (np.cosh(beta[i] * J)) ** -2).sum()
    S[i] = (U[i] - F[i]) / T[i]

F = F / N
U = U / N
C = C / N
S = S / N

fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'hspace': 0.25, 'wspace': 0.25})
axs[0, 0].plot(T, F)
axs[0, 0].set_title("Free Energy")
axs[0, 0].set(xlabel="T", ylabel="F / N")

axs[0, 1].plot(T, U)
axs[0, 1].set_title("Average Energy")
axs[0, 1].set(xlabel="T", ylabel="U / N")

axs[1, 0].plot(T, C)
axs[1, 0].set_title("Heat Capacity")
axs[1, 0].set(xlabel="T", ylabel="C / N")

axs[1, 1].plot(T, S)
axs[1, 1].set_title("Entropy")
axs[1, 1].set(xlabel="T", ylabel="S / N")

plt.show()
