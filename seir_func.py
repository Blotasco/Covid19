import matplotlib.pyplot as plt
import numpy as np
from ODEsolver import *
def SEIR(u,t):
    beta = 0.5; r_ia = 0.1; r_e2=1.25
    lmbda_1=0.33; lmbda_2=0.5; p_a=0.4; mu=0.2

    S, E1, E2, I, Ia, R = u
    N = sum(u)
    dS = -beta*S*I/N - r_ia*beta*S*Ia/N - r_e2*beta*S*E2/N
    dE1 = beta * S * I / N + r_ia * beta * S * Ia / N + r_e2 * beta * S * E2 / N - lmbda_1 * E1
    dE2 = lmbda_1 * (1 - p_a) * E1 - lmbda_2 * E2
    dI = lmbda_2 * E2 - mu * I
    dIa = lmbda_1 * p_a * E1 - mu * Ia
    dR = mu * (I + Ia)
    return [dS, dE1, dE2, dI, dIa, dR]


def test_SEIR():
    computed = SEIR([1]*6,0)
    expected = [-0.19583333333333333, -0.13416666666666668, -0.302, 0.3, -0.068, 0.4]
    assert expected == computed ,"i python hvis du bruker ikke tol i et boolean value da automatisk brukes et tol 1e-16"

def solve_SEIR(T,dt,S_0,E2_0):
    solver = RungeKutta4(SEIR)
    solver.set_initial_condition([S_0, 0, E2_0, 0, 0, 0])
    time_points = np.linspace(0,T,int(T/dt))
    u,t = solver.solve(time_points)
    return u,t

def plot_SEIR(u,t):
    S = u[:,0]; I = u[:,3]; Ia = u[:,4]; R = u[:,5]
    plt.plot(t,S, label = "S")
    plt.plot(t,I, label = "I")
    plt.plot(t,Ia, label = "Ia")
    plt.plot(t,R, label = "R")
    plt.legend()
    plt.show()

u,t = solve_SEIR(100,1.0,5e6,100)
print(plot_SEIR(u,t))


#Kjøreeksempel:
"""
$ python seir_func.py
"""

