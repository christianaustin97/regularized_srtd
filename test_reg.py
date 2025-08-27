import numpy as np
from fenics import errornorm
import oldroyd_3_regularized_SRTD
import matplotlib.pyplot as plt


h=0.025 
rad = 0.5
ecc = 0.25
s = 1.0
eta = 1.0
l1 = 1e-2
a = 1.0 # UCM model
mu1 = a*l1
max_iters = 20
tol = 1e-8

#unregularized SRTD solution
print("Solving the unregularized solution")
print("\n ========================================================================================================================== \n")
soln_jb_unreg = oldroyd_3_regularized_SRTD.oldroyd_3_JB_reg_SRTD(h, rad, ecc, s, eta, l1, mu1, max_iters, tol, 0.0)

u_unreg = soln_jb_unreg.velocity
p_unreg = soln_jb_unreg.pressure
T_unreg = soln_jb_unreg.stress_tensor_vec


epsilon_vals = 10**(np.linspace(-12, -1, 12, endpoint=True))
n = len(epsilon_vals)
u_l2_diff = np.zeros(n)
u_h1_diff = np.zeros(n)
p_l2_diff = np.zeros(n)
p_h1_diff = np.zeros(n)
T_l2_diff = np.zeros(n)
T_h1_diff = np.zeros(n)

for i in range(n):
    epsilon = float(epsilon_vals[i])
    
    print("\n ========================================================================================================================== \n")
    print("solving the regularized solution with eps = %1.3e"%(epsilon))
    soln_jb_reg = oldroyd_3_regularized_SRTD.oldroyd_3_JB_reg_SRTD(h, rad, ecc, s, eta, l1, mu1, max_iters, tol, epsilon)
    u_reg = soln_jb_reg.velocity
    p_reg = soln_jb_reg.pressure
    T_reg = soln_jb_reg.stress_tensor_vec

    u_l2_diff[i] = errornorm(u_unreg, u_reg, "l2")
    u_h1_diff[i] = errornorm(u_unreg, u_reg, "h1")
    p_l2_diff[i] = errornorm(p_unreg, p_reg, "l2")
    p_h1_diff[i] = errornorm(p_unreg, p_reg, "h1")
    T_l2_diff[i] = errornorm(T_unreg, T_reg, "l2")
    T_h1_diff[i] = errornorm(T_unreg, T_reg, "h1")

plt.figure()
plt.title("Journal Bearing Problem")

plt.subplot(2,3,1)
plt.loglog(epsilon_vals, u_l2_diff)
plt.title("$\mathbf{u}$ $L_{2}$ difference")
plt.xlabel("Regularization parameter epsilon")
plt.ylabel("$\mathbf{u}$ $ L_{2}$ difference")

plt.subplot(2,3,2)
plt.loglog(epsilon_vals, p_l2_diff)
plt.title("$p$ $ L_{2}$ difference")
plt.xlabel("Regularization parameter epsilon")
plt.ylabel("$p$ $ L_{2}$ difference")

plt.subplot(2,3,3)
plt.loglog(epsilon_vals, T_l2_diff)
plt.title("$\mathbf{T}$ $L_{2}$ difference")
plt.xlabel("Regularization parameter epsilon")
plt.ylabel("$\mathbf{T}$ $L_{2}$ difference")


plt.subplot(2,3,4)
plt.loglog(epsilon_vals, u_h1_diff)
plt.title("$\mathbf{u}$ $ H^{1}$ difference")
plt.xlabel("Regularization parameter epsilon")
plt.ylabel("$\mathbf{u}$ $ H^{1}$ difference")

plt.subplot(2,3,5)
plt.loglog(epsilon_vals, p_h1_diff)
plt.title("$p$ $ H^{1}$ difference")
plt.xlabel("Regularization parameter epsilon")
plt.ylabel("$p$ $ H^{1}$ difference")

plt.subplot(2,3,6)
plt.loglog(epsilon_vals, T_h1_diff)
plt.title("$\mathbf{T}$ $ H^{1}$ difference")
plt.xlabel("Regularization parameter epsilon")
plt.ylabel("$\mathbf{T}$ $ H^{1}$ difference")

plt.show()







