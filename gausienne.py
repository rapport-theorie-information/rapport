import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt

n = 10000
mu = 2
sigma = np.sqrt(9)

def realisation(x):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

X = sigma*rnd.randn(n)+mu
nbins = 40
hist, bin_edges = np.histogram(X, bins=nbins, density=True)
hist = hist/(np.sum(hist)*(bin_edges[1]-bin_edges[0]))
plt.figure()
plt.bar(bin_edges[:-1], hist, align='edge', width=bin_edges[1] - bin_edges[0])

t = np.linspace(-15,15,1000)
y = [realisation(x) for x in t]
plt.plot(t,y,color='red')
plt.show()
+36
def entropie_numérique_plus_log(X):
    s=0
    for i in range(n):
        s-=realisation(X[i])*np.log(realisation(X[i]))
    s*=1/n
    return s
s=entropie_numérique_plus_log(X)
print(s)
