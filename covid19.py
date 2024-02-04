from SEIR_interaction import *
#a
def area(filename):
    infile = open(filename,"r")
    regionS = []
    for line in infile:
        word = line.replace("\t"," ").replace("\n"," ").strip(" ").split(";")
        name = word[1]
        S_0 = float(word[2])
        E2_0 = float(word[3])
        latitude = float(word[4])
        longitude = float(word[5])
        regionS.append(RegionInteraction(name, S_0, E2_0, latitude=latitude, longitude=longitude))
    infile.close()
    return regionS



regionS = area("fylker.txt")
print(regionS)
#b
def covid19_Norway(beta, filename, num_days, dt):
    regionS = area(filename)
    plt.figure(figsize=(9, 12)) # set figsize
    problem = ProblemInteraction(regionS, area_name=filename , beta= beta)
    problem.set_initial_condition()
    solver = SolverSEIR(problem, T=num_days, dt=dt)
    solver.solve()
    for i, r in enumerate(problem.region,start=1):
        plt.subplot(4,3,i)
        r.plot()
    plt.subplot(4,3, len(regionS)+1)
    problem.plot()
    plt.subplots_adjust(hspace = 0.75, wspace=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    covid19_Norway(beta = 0.5,filename= "fylker.txt",num_days=100,dt = 1)

#Kjøreeksempel
"""
$ python covid19.py
[<SEIR_interaction.RegionInteraction object at 0x0110A2B0>, <SEIR_interaction.RegionInteraction object at 0x0110A2F8>, <SEIR_interaction.RegionInteraction object at 0x01130C40>, <SEIR_interaction.RegionInteraction object at 0x011303D0>, <SEIR_interaction.RegionInteraction object at 0x01130508>, <SEIR_interaction.RegionInteraction object at 0x01130790>, <SEIR_interaction.RegionInteraction object at 0x011303B8>, <SEIR_interaction.RegionInteraction object at 0x01130FD0>, <SEIR_interaction.RegionInteraction object at 0x011303A0>, <SEIR_interaction.RegionInteraction object at 0x01130FB8>, <SEIR_interaction.RegionInteraction object at 0x01150028>]

"""
