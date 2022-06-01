import math
import random

from NEAT.Genome import Genome


class Species:

    def __init__(self, representative):

        self.clients = []

        self.representative = representative
        representative.species = self
        self.clients.append(representative)

        self.score = 0

    
    def put(self, client, distance_max = 3.0):
        if self.representative.distance(client) <= distance_max:
            client.species = self
            self.clients.append(client)

            return True
        
        return False


    def forcePut(self, client):
        client.species = self
        self.clients.append(client)

    
    def goExtinct(self):
        for client in self.clients:
            client.species = None


    def evaluateScore(self):
        result = 0

        for client in self.clients:
            client.adjusted_score = client.score/len(self.clients)
            result += client.adjusted_score

        self.score = result
        return self.score
        
    
    def reset(self):
        if len(self.clients) <= 0:
            return

        self.representative = random.choice(self.clients)

        for client in self.clients:
            client.species = None

        self.clients.clear()

        self.clients.append(self.representative)
        self.representative.species = self
        self.score = 0

    
    def kill(self, percentage = 0.2):
        self.clients.sort(key=lambda x: x.score)

        amount = math.floor(percentage * len(self.clients))
        for i in range(amount):
            self.clients[0].species = None
            self.clients.pop(0)

    
    def breed(self):
        parentA = random.choice(self.clients)
        parentB = random.choice(self.clients)

        if parentA.score > parentB.score:
            return Genome.crossover(parentA.genome, parentB.genome)
        else:
            return Genome.crossover(parentB.genome, parentA.genome)