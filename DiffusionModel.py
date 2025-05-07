"""A 1D diffusion model."""

import numpy as np
import matplotlib.pyplot as plt


# start by setting two fixed model parameters, the diffusivity and the size of the model domain.

# In[ ]:


D = 110
LX = 300


# Next, set up the model grid using NumPy array.

# In[ ]:


dx = 0.5
x = np.arange(start=0, stop=LX, step=dx)
nx = len(x)


# set the initial conditions for the model.
# The cake  `C` is a step function with a high value of the left, a low value on the right, and a step at the center of the domain.

# In[ ]:


C = np.zeros_like(x)
C_left = 500
C_right = 0
C[x <= LX / 2] = C_left
C[x > LX / 2] = C_right


# plot the initial profile.

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("X")
plt.ylabel("C")
plt.title("Initial profile")


# Set the number of time steps in the model.
# Calculate a stable time step using a stablity criterion.

# In[ ]:


nt = 5000
dt = 0.5 * dx ** 2 / D


# loop over the time steps of the model, solving the diffusion equation using FTCS scmeg shown above.
# Note trge use of array operations on the varianle `C. the boundry conditions rmain fixed in each time step.

# In[ ]:


for t in range(0,nt):
    C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])


# plot the reult.

# In[ ]:


plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final Profile")


# In[ ]:




