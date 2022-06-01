import random
import numpy

from NEAT.Genome import Genome
from NEAT.Neat import Neat

inputs = 2
output = 1

population = 1000

data = [[[0, 0], 0], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]]


neat = Neat(inputs, output, population)


for i in range(100):
    print("-------------------------------------- Generation", i+1, "--------------------------------------")

    for client in neat.clients:
        error = 0
        for j in range(len(data)):
            answer = client.calculate(data[j][0])[0]
            error += numpy.absolute(data[j][1] - answer)
    
        client.score = (len(data) - error) ** 2

    neat.evolve()

    neat.species.sort(key=lambda x: x.score, reverse=True)
    neat.printSpecies()


max_score = -1
max_species = None

for species in neat.species:
    if species.score > max_score:
        max_score = species.score
        max_species = species

max_score = -1
max_client = None
for client in max_species.clients:
        if client.score > max_score:
            max_score = client.score
            max_client = client

max_client.genome.nodes.sort(key=lambda x: (x.layer, x.innovation_number))
max_client.genome.connections.sort(key=lambda x: (x.OUT_node.layer, x.IN_node.layer))

for node in max_client.genome.nodes:
    print(node)
for connection in max_client.genome.connections:
    print(connection)

print(max_client.calculate(data[0][0])[0])
print(max_client.calculate(data[1][0])[0])
print(max_client.calculate(data[2][0])[0])
print(max_client.calculate(data[3][0])[0])