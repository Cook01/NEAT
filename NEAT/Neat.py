import math
import random

from NEAT.ConnectionGene import ConnectionGene
from NEAT.Genome import Genome
from NEAT.NodeGene import NodeGene
from NEAT.Client import Client
from NEAT.Species import Species

# The Global state of the NEAT Algorithm
class Neat:

#--------------------------------------------------------- Global Variables ---------------------------------------------------------

    # Nb Nodes max (~1 000 000)
    MAX_NODES = math.pow(2, 20)

#--------------------------------------------------------- Utils ---------------------------------------------------------

    def __init__(self, input_size, output_size, nb_clients):
        # Global list of existing Nodes
        self.nodes_list = []
        # Global list of existing Connections
        self.connections_list = []

        self.nb_clients = nb_clients

        self.clients = []
        self.species = []

        # Nb of Inputs
        self.input_size = input_size
        # Nb of Outputs
        self.output_size = output_size

        # Reset (initiate) the global NEAT
        self.reset(input_size, output_size, nb_clients)


    # Reset the global NEAT
    def reset(self, input_size, output_size, nb_clients):
        # Reset Inputs size
        self.input_size = input_size
        # Reset Outputs size
        self.output_size = output_size

        self.nb_clients = nb_clients

        # Clear the global list of existing Nodes
        self.nodes_list.clear()
        # Clear the global list of existing Connections
        self.connections_list.clear()

        self.clients.clear()

        # Create Inputs nodes
        for i in range(self.input_size):
            new_node = self.createNewNode()
            new_node.layer = 0

        # Create Outputs nodes
        for i in range(self.output_size):
            new_node = self.createNewNode()
            new_node.layer = 1
        
        for i in range(nb_clients):
            self.clients.append(Client(self.createEmptyGenome()))

#--------------------------------------------------------- Genome ---------------------------------------------------------

    # Create a new empty Genome
    def createEmptyGenome(self):
        # Create new Genome
        new_genome = Genome(self)

        # Create Inputs and Outputs
        for i in range(self.input_size + self.output_size):
            new_genome.nodes.append(self.getNode(i + 1))

        # Return the new Genome
        return new_genome

#--------------------------------------------------------- Node ---------------------------------------------------------

    # Create a new Node
    def createNewNode(self):
        # Create new Node (with a new innovation number)
        new_node = NodeGene(len(self.nodes_list) + 1)
        # Add new Node to global Nodes List
        self.nodes_list.append(new_node)

        # Return new Node
        return new_node

    
    # Get a Node by its innovation number
    def getNode(self, innovation_number):
        # if innovation number is in the list
        if innovation_number <= len(self.nodes_list):
            # Return corresponding Node
            return self.nodes_list[innovation_number - 1]

        # Else, create and return a new Node
        return self.createNewNode()

#--------------------------------------------------------- Connection ---------------------------------------------------------

    # Try to create a new Connection
    def newConnection(self, IN_node, OUT_node):
        # Create new Connection (with tmp innovation number)
        new_connection = ConnectionGene(-1, IN_node, OUT_node)

        # If Connection already exist in the global Connections List
        if new_connection in self.connections_list:
            index = self.connections_list.index(new_connection)
            # Set innovation number as the same as the one that already exist
            new_connection.innovation_number = self.connections_list[index].innovation_number
        else:
            # Else, set a new innovation number
            new_connection.innovation_number = len(self.connections_list) + 1
            # And add the new Connection to the global Connections List
            self.connections_list.append(new_connection)

        # Return new Connection
        return new_connection


    def getReplaceIndex(self, node_in, node_out):
        connection = ConnectionGene(-1, node_in, node_out)

        if connection in self.connections_list:
            return self.connections_list[self.connections_list.index(connection)].replace_index
        else:
            return -1

        
    def setReplaceIndex(self, node_in, node_out, index):
        connection = ConnectionGene(-1, node_in, node_out)

        if connection in self.connections_list:
            self.connections_list[self.connections_list.index(connection)].replace_index = index


    def evolve(self):
        self.generateSpecies()

        self.kill()
        self.removeExtinct()

        self.reproduce()
        self.mutate()

        for client in self.clients:
            client.generateCalculator()

    

    def generateSpecies(self):
        for species in self.species:
            species.reset()

        for client in self.clients:
            if client.species != None:
                continue
            
            found = False
            for species in self.species:
                if species.put(client):
                    found = True
                    break
            
            if not found:
                self.species.append(Species(client))


        for species in self.species:
            species.evaluateScore()


    def kill(self):
        max_score = -1
        min_score = math.inf

        for species in self.species:
            if species.score > max_score:
                max_score = species.score
            if species.score < min_score:
                min_score = species.score

        for species in self.species:
            if max_score == min_score:
                kill_rate = 1
            else:
                kill_rate = 1 - ((species.score - min_score) / (max_score - min_score))

            species.kill(0.8 * kill_rate)

        

    
    def removeExtinct(self):
        amount = len(self.species) - 1
        for i in range(amount, -1, -1):
            if len(self.species[i].clients) <= 1:
                self.species[i].goExtinct()
                self.species.pop(i)


    def reproduce(self):
        for client in self.clients:
            if client.species != None:
                continue
            
            weights = []
            for species in self.species:
                weights.append(species.score)
            random_species = random.choices(self.species, weights=weights, k=1)[0]

            client.genome = random_species.breed()
            random_species.forcePut(client)


    def mutate(self):
        for client in self.clients:
            client.mutate()





    def printSpecies(self):
        print("################################################")
        for species in self.species:
            print(species, species.score, len(species.clients))