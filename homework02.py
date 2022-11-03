import numpy as np
from matplotlib import pyplot as plt

# Generate random numbers between 0 and 1
x = np.random.rand(100)

# Calculating targets
t = [x**3 - x**2 for x in x]

# Plotting the data points
plt.scatter(x, t, s=5)
plt.show()

