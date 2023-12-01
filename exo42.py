import numpy as np
from numpy import random as rnd
import scipy.linalg as la
import scipy.io as sio
import matplotlib.pyplot as plt

#Question 1:
mat_contents = sio.loadmat("pluv.mat")
bordeaux=[x[0] for x in mat_contents["X_pluv"]]
nantes=[x[1] for x in mat_contents["X_pluv"]]
santiago=[x[2] for x in mat_contents["X_pluv"]]

#calcul des Xk:
X_b_n = np.array([[bordeaux[i], nantes[i]] for i in range(len(bordeaux))])
X_b_s = np.array([[bordeaux[i], santiago[i]] for i in range(len(bordeaux))])
X_n_s = np.array([[nantes[i], santiago[i]] for i in range(len(nantes))])

# Calcul des moyennes emperiques:
mu_b_n = np.mean(X_b_n, axis=0)
mu_b_s = np.mean(X_b_s, axis=0)
mu_n_s = np.mean(X_n_s, axis=0)


#Calcul de R:
n = len(X_b_n)
R_b_n = (1/n) * sum([(x - mu_b_n).reshape(-1, 1) @ (x - mu_b_n).reshape(1, -1) for x in X_b_n])
R_b_s = (1/n) * sum([(x - mu_b_s).reshape(-1, 1) @ (x - mu_b_s).reshape(1, -1) for x in X_b_s])
R_n_s = (1/n) * sum([(x - mu_n_s).reshape(-1, 1) @ (x - mu_n_s).reshape(1, -1) for x in X_n_s])


#tracer les lignes:
step = 10**(-3)
theta = np.arange(0, 2*np.pi, step)
w = np.array([np.cos(theta), np.sin(theta)])

x1 = la.sqrtm(R_b_n) @ w + mu_b_n.reshape(-1, 1)
x2 = la.sqrtm(R_b_s) @ w + mu_b_s.reshape(-1, 1)
x3 = la.sqrtm(R_n_s) @ w + mu_n_s.reshape(-1, 1)




fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Scatter plot for Bordeaux-Nantes
axes[0].plot(x1[0], x1[1], color='red')
axes[0].scatter(X_b_n[:, 0], X_b_n[:, 1], c='blue', label='Bordeaux-Nantes Data')
axes[0].legend()
axes[0].set_xlabel('Bordeaux')
axes[0].set_ylabel('Nantes')


# Scatter plot for Bordeaux-Santiago
axes[1].plot(x2[0], x2[1], color='green')
axes[1].scatter(X_b_s[:, 0], X_b_s[:, 1], c='yellow', label='Bordeaux-Santiago Data')
axes[1].legend()
axes[1].set_xlabel('Bordeaux')
axes[1].set_ylabel('Santiago')

# Scatter plot for Nantes-Santiago
axes[2].plot(x3[0], x3[1], color='black')
axes[2].scatter(X_n_s[:, 0], X_n_s[:, 1], c='green', label='Nantes-Santiago Data')
axes[2].legend()
axes[2].set_xlabel('Nantes')
axes[2].set_ylabel('Santiago')


plt.tight_layout()
plt.show()
