from audioop import bias
import math
import random
from NEAT.Gene import Gene

# Genes that represent a node
class NodeGene(Gene):

    def __init__(self, innovation_number):
        super().__init__(innovation_number)
        self.layer = -1

        self.output = 0.0
        self.connections = []

        self.bias = 0

    def __eq__(self, other):
        if type(self) != type(other): return False

        return self.innovation_number == other.innovation_number


    def __repr__(self):
       return str(self.innovation_number) + " (" + str(self.layer) + ")"


    def __str__(self):
       return str(self.innovation_number) + " (" + str(self.layer) + ")"


#--------------------------------------------------------- Calculate ---------------------------------------------------------

    def calculate(self):
        sum = 0

        for connection in self.connections:
            if connection.enabled:
                sum += connection.weight * connection.IN_node.output

        sum += self.bias

        self.output = NodeGene.activation(sum)
        return self.output


    @staticmethod
    def activation(x):
        return 1/(1 + math.exp(-4.9 * x))