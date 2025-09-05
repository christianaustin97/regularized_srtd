"""
Tests if SRTD and/or EVSS can handle Poisueille flow, or simple geometry with inflow

As of now, February 2, 2025, the following only works for the UCM and LCM (a=1 and -1, respectively)
This is because the flow profile is only parabolic for the UCM and LCM, and more complicated for 
anything in between, a \in (-1,1). 

This version uses a 3-dimensional VectorElement for the stress tensor, then reshapes it to look like
a tensor. This is because using symmetric tensors is reportedly buggy


"""
from fenics import *
import matplotlib.pyplot as plt

nx = 24
#mesh = UnitSquareMesh(nx, nx, "crossed")
width = 4.0
mesh = RectangleMesh(Point(0,0), Point(width,1), 4*nx, nx, "crossed")

max_iter = 20
tol = 1e-10

a = 1.0 # a=1 ucm, a=-1 lcm
eta_0 = 1.0
l1 = 1e-2 
mu1 = a*l1

C = 5.0 # pressure gradient 
D = (1.0-a)*l1*C*C/(4.0*eta_0) # arbitrary constant defining pressure

#h= C /(8.0*nx) # for SUPG, characteristic (max) velocity = C/16, times meshsize (extra factor of 2 because "crossed")
h=0.0 # unfortunately SUPG seems to make it worse lol

# body forces
f = Constant((0.0, 0.0)) # no body forces


# Element spaces
P_elem = FiniteElement("CG", triangle, 1) #pressure and auxiliary pressure, degree 1 elements
V_elem = VectorElement("CG", triangle, 2) #velocity, degree 2 elements
T_elem = VectorElement("CG", triangle, 2, dim=3) #stress tensor has 3 components, because it is symmetric. Symmetric stress tensors are apparently buggy
    
W_elem = MixedElement([V_elem, P_elem]) # Mixed/Taylor Hood element space for Navier-Stokes type equations

W = FunctionSpace(mesh, W_elem) # Taylor-Hood/mixed space
P = FunctionSpace(mesh, P_elem) # true pressure space
V = FunctionSpace(mesh, V_elem) # velocity space (not used)
T = FunctionSpace(mesh, T_elem) # tensor space

# boundary conditions. Also, all of these should be the same, so we can use these for error checking
u_1_in = Expression("C*x[1]*(1-x[1])/(2*eta)", C=C, eta=eta_0, degree=2)
u_in = Expression(("u", "0.0"), u=u_1_in, degree = 2)
u_walls = Constant((0.0, 0.0))

T_11 = Expression("(a+1)*l1*C*C*(2*x[1]-1)*(2*x[1]-1)/(4*eta)", a=a, l1=l1, C=C, eta=eta_0, degree=2)
T_12 = Expression("-C*(2*x[1]-1)/2", C=C, degree=2)
T_22 = Expression("(a-1)*l1*C*C*(2*x[1]-1)*(2*x[1]-1)/(4*eta)", a=a, l1=l1, C=C, eta=eta_0, degree=2)
#T_22 = Expression("0.0", degree=2)
T_inlet = Expression(("T_11", "T_12", "T_22"), T_11=T_11, T_12=T_12, T_22=T_22, degree=2)


# Interpolate body force and BCs onto velocity FE space
u_in = interpolate(u_in, W.sub(0).collapse()) 
u_walls = interpolate(u_walls, W.sub(0).collapse()) 

# inflow Poiseuille-like stress tensor, interpolate onto stress tensor space
T_inlet_vec = interpolate(T_inlet, T) 
#T_inlet = as_tensor([[T_inlet_vec[0], T_inlet_vec[1]], [T_inlet_vec[1], T_inlet_vec[2]]])
# body force
f = interpolate(f, W.sub(0).collapse())

# Define key boundaries
class Inlet(SubDomain):
        def inside(self, x, on_boundary):
            return (near(x[0], 0.0) and on_boundary)
inlet = 'near(x[0], 0.0) && on_boundary'

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[0], 4.0) and on_boundary)
outlet = 'near(x[0], 4.0) && on_boundary' 

class Corner(SubDomain):
        # for pressure regulating. DO NOT include "and on_boundary". idk why it just breaks everything
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and near(x[1], 1.0)
corner = 'near(x[0], 0.0) && near(x[1], 1.0)' #for pressure regulating

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return ((near(x[1], 0.0) or near(x[1], 1.0)) and on_boundary )
        #return ((x[1]< 0.0+1.0/24.0) or (x[1] > 1.0-1.0/24.0)) and on_boundary
        #return (not (near(x[0], 4.0) or near(x[0], 0.0))) and on_boundary
        # First two appear to be equivalent. Last one DOES NOT work, not sure why
walls = '(near(x[1], 0.0) || near(x[1], 1.0)) && on_boundary'


bcu_inflow = DirichletBC(W.sub(0), u_in, Inlet())
bcu_outflow = DirichletBC(W.sub(0), u_in, Outlet())
bcu_walls = DirichletBC(W.sub(0), u_walls, Walls())
aux_pressure_reg = DirichletBC(W.sub(1), Constant(0.0), Corner(), 'pointwise')
bcu = [bcu_inflow, bcu_outflow, bcu_walls, aux_pressure_reg]
#bcu = [bcu_inflow, bcu_outflow, bcu_walls]

p_inlet = Expression("T22 - C*x[0]+D", T22 = T_22, C=C, D = D, degree=2)
p_inlet = interpolate(p_inlet, P)

bcp = [DirichletBC(P, p_inlet, Inlet())]

bcT_inletflow = DirichletBC(T, T_inlet, Inlet())
bcT = [bcT_inletflow] # don't think this is necessary for one BC

# Variational Problem Begin
#
# Trial Functions. Think of TrialFunctions as symbolic, and they are only used in defining the weak forms
w = TrialFunction(W) # our NS-like TrialFunction
(u,pi) = split(w) # trial functions, representing u1, pi1
p = TrialFunction(P) # true pressure trial function for auxiliary pressure equation, representing p1
tau_vec = TrialFunction(T) # stress trial function for stress tensor equation, representing T1
tau = as_tensor([[tau_vec[0], tau_vec[1]], [tau_vec[1], tau_vec[2]]])

# Weak form test functions. Also think of these as symbolic, and they are only used in defining the weak forms
(v, q) = TestFunctions(W) # test functions for NSE step
r = TestFunction(P) # test functions for pressure transport
S_vec = TestFunction(T) # test functions for constitutive equation
S = as_tensor([[S_vec[0], S_vec[1]], [S_vec[1], S_vec[2]]])

# previous and next iterations. Symbolic when they are used in the weak forms, or pointers to the actual function values 
#w0 = Function(W)
u0 = Function(V)    
#pi0 = Function(P)
p0 = Function(P)
T0_vec = Function(T)
T0 = as_tensor([[T0_vec[0], T0_vec[1]], [T0_vec[1], T0_vec[2]]])

w1 = Function(W)
u1 = Function(V)
pi1 = Function(P)
p1 = Function(P)
T1_vec = Function(T) # do not reshape, as FEniCS is expecting T1 to be a 3-dimensional vector

# Functions we'll actually return
u_return = Function(V)
pi_return = Function(P)
p_return = Function(P)
T_return_vec = Function(T)

#LHS of NS-like solve, a((u,pi), (v,q)) 
a_nse = eta_0*inner(grad(u), grad(v))*dx + dot( dot(grad(u),u), v)*dx - (pi*div(v))*dx + q*div(u)*dx

# RHS of NS-like stage is given in section 7 of Girault/Scott paper F((v,q); u0, T0)
term1 = inner(f, v - l1*dot(grad(v), u0))*dx #orange term
term2 = (p0*inner(nabla_grad(u0), grad(v)))*dx  #blue term
term3 = -inner( dot(grad(u0),u0) , dot(grad(v),u0) )*dx #red term
term4 = inner( dot(grad(u0),T0) , grad(v) )*dx #light green term
term5 = inner( dot(sym(grad(u0)),T0)+dot(T0,sym(grad(u0))) , grad(v) )*dx #dark green term

L_nse = term1 - l1*(term2 + term3 + term4) + (l1-mu1)*term5 #mathcal F 

# Nonlinear in u, so must solve a-L==0 and use Newton instead of a==L directly
F = a_nse - L_nse

# Nonlinear NSE, so using Newton iteration
F_act = action(F, w1) 
dF = derivative(F_act, w1)
nse_problem = NonlinearVariationalProblem(F_act, w1, bcu, dF) # will update w1 values every time we call solver.solve()
nse_solver = NonlinearVariationalSolver(nse_problem)
nse_prm = nse_solver.parameters
nse_prm["nonlinear_solver"] = "newton"
#nse_prm["newton_solver"]["linear_solver"] = "mumps" # utilizes parallel processors

# Pressure transport equation with SUPG
ap = (p + l1*dot(grad(p), u1)) * r * dx 
Lp = pi1 * r * dx 

p_problem = LinearVariationalProblem(ap, Lp, p1, bcs = bcp) # will update p1 values every time we call solver.solve()
p_solver = LinearVariationalSolver(p_problem)


# Stress transport equation/Constitutive equation with SUPG
aT = inner( tau + l1*(dot(grad(tau),u1) + dot(-skew(grad(u1)), tau) - dot(tau, -skew(grad(u1)))) \
                    - mu1*(dot(sym(grad(u1)), tau) + dot(tau, sym(grad(u1)))) , S)*dx
LT = 2.0*eta_0*inner(sym(grad(u1)), S)*dx

T_problem = LinearVariationalProblem(aT, LT, T1_vec, bcs=bcT) # will update p1 values every time we call solver.solve()
T_solver = LinearVariationalSolver(T_problem)
T_prm = T_solver.parameters
#T_prm["linear_solver"] = "mumps"

# Begin SRTD iterative solve
n=1
l2diff = 1.0
residuals = {} # empty dict to save residual value after each iteration 
Newton_iters = {}
min_residual = 1.0
while(n<=max_iter and min_residual > tol):
    try: 
        (Newton_iters[n], converged) = nse_solver.solve() # updates w1
    except: 
        print("Newton Method in the Navier-Stokes-like stage failed to converge")
    
    u_next, pi_next = w1.split(deepcopy=True)
    assign(u1, u_next) # u1 updated
    assign(pi1, pi_next) # pi1 updated

    p_solver.solve() # p1 updated

    T_solver.solve() # T1_vec updated

    # End of this SRTD iteration
    l2diff = errornorm(u1, u0, norm_type='l2', degree_rise=0)
    residuals[n] = l2diff
    if(l2diff <= min_residual):
        min_residual = l2diff
        u_return = u1
        pi_return = pi1
        p_return = p1
        T_return_vec = T1_vec
        T_return = as_tensor([[T1_vec[0], T1_vec[1]], [T1_vec[1], T1_vec[2]]])

    print("SRTD Iteration %d: r = %.4e (tol = %.3e)" % (n, l2diff, tol))
    n = n+1
    
    #update u0, p0, T0
    assign(u0, u1)
    assign(p0, p1)
    assign(T0_vec, T1_vec) 
    T0 = as_tensor([[T0_vec[0], T0_vec[1]], [T0_vec[1], T0_vec[2]]]) # maybe unnecessary?
    
# Stuff to do after the iterations are over
T1 = as_tensor([[T1_vec[0], T1_vec[1]], [T1_vec[1], T1_vec[2]]])
T_inlet = as_tensor([[T_inlet_vec[0], T_inlet_vec[1]], [T_inlet_vec[1], T_inlet_vec[2]]])

if(min_residual <= tol):
    converged = True
else:
    converged = False

u_diff = u1 - u_in
print("u l2 error = %1.4e"%errornorm(u1, u_in, "l2"))
print("u h1 error = %1.4e"%errornorm(u1, u_in, "h1"))
print("T l2 error = %1.4e"%errornorm(T1_vec, T_inlet_vec, "l2"))

plt.subplot(3,3,1)
fig2 = plot(sqrt(inner(u1, u1)))
plt.title("Velocity Magnitude")
plt.colorbar(fig2, label = "magnitude")

plt.subplot(3,3,2)
fig = plot(pi1)
plt.title("Aux Pressure")
plt.colorbar(fig, label = "magnitude")

plt.subplot(3,3,3)
fig3 = plot(p1)
plt.title("Pressure")
plt.colorbar(fig3)

plt.subplot(3,3,4)
fig4 = plot(T1[0,0])
plt.title("T_11")
plt.colorbar(fig4)

plt.subplot(3,3,5)
fig5 = plot(T1[0,1])
plt.title("T_12")
plt.colorbar(fig5)

plt.subplot(3,3,6)
fig6 = plot(T1[1,1])
plt.title("T_22")
plt.colorbar(fig6)

plt.subplot(3,3,7)
fig7 = plot(T_inlet[0,0])
plt.title("T_11 True")
plt.colorbar(fig7)

plt.subplot(3,3,8)
fig8 = plot(T_inlet[0,1])
plt.title("T_12 True")
plt.colorbar(fig8)

plt.subplot(3,3,9)
fig9 = plot(T_inlet[1,1])
plt.title("T_22 True")
plt.colorbar(fig9)

#plt.savefig("inflow_srtd_as_tensor.pdf", bbox_inches = 'tight')


plt.show()




