from NEAT.Calculator import Calculator
from NEAT.Genome import Genome


class Client:

    def __init__(self, genome = None):
        self.calculator = None

        self.genome = genome
        self.score = 0
        self.adjusted_score = 0
        self.species = None


    def generateCalculator(self):
        self.calculator = Calculator(self.genome)


    def calculate(self, input_list):
        if self.calculator == None:
            self.generateCalculator()

        return self.calculator.calculate(input_list)


    def distance(self, other_client):
        return Genome.distance(self.genome, other_client.genome)


    def mutate(self):
        self.genome.mutate()