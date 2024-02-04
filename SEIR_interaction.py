from SEIR import *
import numpy as np
class RegionInteraction(Region):
    def __init__(self,*args,latitude, longitude, **kwargs):
        super().__init__(*args, **kwargs)
        self.latitude = latitude*np.pi/180
        self.longitude = longitude*np.pi/180

    def distance(self,other):
        R = 64
        self.latitude = self.latitude
        other.latitude = other.latitude
        self.longitude = self.longitude
        other.longitude = other.longitude
        inner = np.sin(self.latitude)*np.sin(other.latitude)+np.cos(self.latitude)*np.cos(other.latitude)*np.cos(abs(self.longitude-other.longitude))
        d = R * np.arccos(np.clip(inner, 0, 1))
        return d


if __name__ == "__main__":
    innlandet = RegionInteraction("Innlandet",S_0=371385, E2_0=0,latitude=60.7945,longitude=11.0680)
    oslo = RegionInteraction("Oslo",S_0=693494,E2_0=100,latitude=59.9,longitude=10.8)
    print(oslo.distance(innlandet))


class ProblemInteraction(ProblemSEIR):
    def __init__(self, region, area_name, beta, r_ia=0.1, r_e2=1.25,lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        self.area_name = area_name
        super().__init__(region, beta, r_ia, r_e2, lmbda_1, lmbda_2, p_a, mu)

    def get_population(self):
        reg = 0
        for i in range(len(self.region)):
            reg += self.region[i].population
        return reg

    def set_initial_condition(self):
        self.initial_condition = []
        for i in range(len(self.region)):
            self.initial_condition += [self.region[i].S_0, self.region[i].E1_0, self.region[i].E2_0, self.region[i].I_0, self.region[i].Ia_0, self.region[i].R_0]
        return self.initial_condition

    def __call__(self, u,t):
        n = len(self.region)
        SEIR_list = [u[i:i+6] for i in range(0, len(u), 6)]
        E2_list = [u[i] for i in range(2, len(u), 6)]
        Ia_list = [u[i] for i in range(4, len(u), 6)]
        derivative = []
        for i in range(n):
            S, E1, E2, I, Ia, R = SEIR_list[i]
            Ni = sum(SEIR_list[i]) #self.region[i].population
            dS = -(self.beta(t) * S * I)/Ni
            for j in range(n):
                E2_other = E2_list[j]
                Ia_other = Ia_list[j]
                Nj = sum(SEIR_list[j]) #self.region[j].population
                dij = self.region[i].distance(self.region[j])
                dS += - (self.r_ia * self.beta(t) * S * Ia_other * np.exp(-dij)/Nj) - (self.r_e2 * self.beta(t) * S * E2_other * np.exp(-dij) / Nj)
            dE1 = -dS -self.lmbda_1 * E1
            dE2 = self.lmbda_1 * (1 - self.p_a) * E1 - self.lmbda_2 * E2
            dI = self.lmbda_2 * E2 - self.mu * I
            dIa = self.lmbda_1 * self.p_a * E1 - self.mu * Ia
            dR = self.mu * (I + Ia)
            derivative += [dS, dE1, dE2, dI, dIa, dR]
        return derivative

    def solution(self, u, t):
        n = len(t)
        n_reg = len(self.region)
        self.t = t
        self.S = np.zeros(n)
        self.E1 = np.zeros(n)
        self.E2 = np.zeros(n)
        self.I = np.zeros(n)
        self.Ia = np.zeros(n)
        self.R = np.zeros(n)
        SEIR_list = [u[:, i:i + 6] for i in range(0, n_reg * 6, 6)]
        for part, SEIR in zip(self.region, SEIR_list):
            part.set_SEIR_values(SEIR, t)
            self.S += u[:, 0]
            self.E1 += u[:, 1]
            self.E2 += u[:, 2]
            self.I += u[:, 3]
            self.Ia += u[:, 4]
            self.R += u[:, 5]

    def plot(self):
        plt.plot(self.t , self.S, label = "Suspectible")
        plt.plot(self.t, self.I, label = "Infected")
        plt.plot(self.t, self.Ia, label = "Infected asymptomatic")
        plt.plot(self.t, self.R, label = "Recovered")
        plt.xlabel("Time (days)")
        plt.ylabel("Population")

if __name__ == "__main__":
    problem = ProblemInteraction([oslo, innlandet],"Norway_east", beta = 0.5)
    print(problem.get_population())
    problem.set_initial_condition()
    print(problem.initial_condition)  # non-nested list of length 12
    u = problem.initial_condition
    print(problem(u, 0))  # list of length 12. Check that values make sense
    # when lines above work, add this code to solve a test problem:
    solver = SolverSEIR(problem, T=100, dt=1.0)
    solver.solve()
    problem.plot()
    plt.legend()
    plt.show()

# Kjøreeksempel:
"""
$ python SEIR_interaction.py
1.0100809386280782
1064979
[693494, 0, 100, 0, 0, 0, 371385, 0, 0, 0, 0, 0]
[-62.49098896472576, 62.49098896472576, -50.0, 50.0, 0.0, 0.0, -12.187832324277785, 12.187832324277785, 0.0, 0.0, 0.0, 0.0]
"""