import random

from NEAT.Calculator import Calculator
from NEAT.ConnectionGene import ConnectionGene

# Represent an instance of a NEAT neural network
class Genome:

    def __init__(self, neat):

        # Connections Genes
        self.connections = []
        # Nodes Genes
        self.nodes = []

        # Global state of the algorithm
        self.neat = neat

        self.calculator = None


#--------------------------------------------------------- Distance ---------------------------------------------------------


    # Calculate the distance between 2 Genomes
    @staticmethod
    def distance(genome_A, genome_B, c1 = 1.0, c2 = 1.0, c3 = 0.4):
        genome_max = genome_A
        genome_min = genome_B

        # Set Genome with Excess Connections as "max"
        highest_innovation_number_A = 0
        highest_innovation_number_B = 0

        if len(genome_A.connections) > 0:
            highest_innovation_number_A = genome_A.connections[-1].innovation_number

        if len(genome_B.connections) > 0:
            highest_innovation_number_B = genome_B.connections[-1].innovation_number

        if highest_innovation_number_A < highest_innovation_number_B:
            genome_max = genome_B
            genome_min = genome_A

        index_A = 0
        index_B = 0


        # Nb of Disjoint Genes
        disjoint = 0
        # Nb of Excess Genes
        excess = 0

        # Average of the Weights deifference
        weight_difference = 0.0
        # Nb of similar Genes
        nb_similar_gene = 0


        # Iterate through both Genome Connections
        while (index_A < len(genome_max.connections)) and (index_B < len(genome_min.connections)):
            connection_A = genome_max.connections[index_A]
            connection_B = genome_min.connections[index_B]

            # If same innovation number : similar Genes
            if connection_A.innovation_number == connection_B.innovation_number :
                nb_similar_gene += 1
                # Tracking weight difference for Average
                weight_difference += abs(connection_A.weight - connection_B.weight)

                index_A += 1
                index_B += 1
            elif connection_A.innovation_number > connection_B.innovation_number:
                # Disjoint Gene of B
                disjoint += 1

                index_B += 1
            else:
                # Disjoint Gene of A
                disjoint += 1

                index_A += 1
            
        # Average weight difference
        if nb_similar_gene == 0:
            nb_similar_gene = 1

        weight_difference /= nb_similar_gene
        # Count excess Genes
        excess = len(genome_max.connections) - index_A

        # Nb connections max
        nb_connections_max = max(len(genome_max.connections), len(genome_min.connections))
        # if N < 20 -> N = 1
        if nb_connections_max < 20:
            nb_connections_max = 1

        # Distance = (Excess / N) + (Desjoint / N) + Weight Diff Average
        return (c1 * (excess / nb_connections_max)) + (c2 * (disjoint / nb_connections_max)) + (c3 * weight_difference)


#--------------------------------------------------------- Crossover ---------------------------------------------------------


    #Crossover 2 Genomes
    #(parent_A has better fitness than parent_B)
    @staticmethod
    def crossover(parent_A, parent_B, selection_rate = 0.5):

        neat = parent_A.neat
        child = neat.createEmptyGenome()

        if len(parent_A.connections) == 0:
            return neat.createEmptyGenome()
        if len(parent_B.connections) == 0:
            return parent_A

        genome_max = parent_A
        genome_min = parent_B
        conversion = False

        # Set Genome with Excess Connections as "max"
        highest_innovation_number_A = 0
        highest_innovation_number_B = 0

        if len(parent_A.connections) > 0:
            highest_innovation_number_A = parent_A.connections[-1].innovation_number

        if len(parent_B.connections) > 0:
            highest_innovation_number_B = parent_B.connections[-1].innovation_number

        if highest_innovation_number_A < highest_innovation_number_B:
            genome_max = parent_B
            genome_min = parent_A
            conversion = True

        index_A = 0
        index_B = 0

        # Iterate through both Genome Connections
        while (index_A < len(genome_max.connections)) and (index_B < len(genome_min.connections)):
            connection_A = genome_max.connections[index_A]
            connection_B = genome_min.connections[index_B]

            new_connection = None

            # If same innovation number : similar Genes
            if connection_A.innovation_number == connection_B.innovation_number :
                # Randomly select one of the parent's Gene
                if(random.random() < selection_rate):
                    new_connection = connection_A.copy()
                else:
                    new_connection = connection_B.copy()

                index_A += 1
                index_B += 1
            elif connection_A.innovation_number > connection_B.innovation_number:
                # Disjoint Gene of B
                if conversion:
                    new_connection = connection_B.copy()

                index_B += 1
            else:
                # Disjoint Gene of A
                if not conversion:
                    new_connection = connection_A.copy()

                index_A += 1

            if new_connection != None:
                if not new_connection.enabled:
                    if random.random() < 0.25:
                        new_connection.enabled = True
            
                child.connections.append(new_connection)


        # Add excess Genes
        if not conversion:
            while index_A < len(genome_max.connections):
                new_connection = genome_max.connections[index_A].copy()
                if not new_connection.enabled:
                    if random.random() < 0.25:
                        new_connection.enabled = True
            
                child.connections.append(new_connection)
                index_A += 1


        # Get Nodes
        # For each connections
        for connection in child.connections:
            IN_exist = False
            OUT_exist = False

            # Check if IN node and OUT node are already in Child
            for node in child.nodes:
                if node == connection.IN_node:
                    IN_exist = True
                if node == connection.OUT_node:
                    OUT_exist = True

            # Add IN node and OUT node
            if not IN_exist:
                child.nodes.append(connection.IN_node)
            if not OUT_exist:
                child.nodes.append(connection.OUT_node)

        child.nodes.sort(key=lambda x: x.innovation_number)
        child.connections.sort(key=lambda x: x.innovation_number)

        # Return new Genome
        return child


#--------------------------------------------------------- Mutation ---------------------------------------------------------


    # def mutate(self, mutate_link_rate = 0.1, mutate_node_rate = 0.1, weight_shift_rate = 0.1, weight_random_rate = 0.1, link_toggle_rate = 0.1, shift_scale = 0.1, random_scale = 1):

    #     for connection in self.connections:
    #         weight_mutation = random.choices([0, 1, 2, 3], weights=[1 - weight_shift_rate - weight_random_rate - link_toggle_rate, weight_shift_rate, weight_random_rate, link_toggle_rate], k = 1)[0]
            
    #         if weight_mutation == 1:
    #             self.mutateWeightShift(connection, shift_scale)
    #         elif weight_mutation == 2:
    #             self.mutateWeightRandom(connection, random_scale)
    #         elif weight_mutation == 3:
    #             self.mutateLinkToggle(connection)

    #     for node in self.nodes:
    #         bias_mutation = random.choices([0, 1, 2], weights=[1 - weight_shift_rate - weight_random_rate, weight_shift_rate, weight_random_rate], k = 1)[0]
            
    #         if bias_mutation == 1:
    #             self.mutateBiasShift(node, shift_scale)
    #         elif bias_mutation == 2:
    #             self.mutateBiasRandom(node, random_scale)


    #     structural_mutation = random.choices([0, 1, 2], weights=[1 - mutate_link_rate - mutate_node_rate, mutate_link_rate, mutate_node_rate], k = 1)[0]
        
    #     if structural_mutation == 1:
    #         self.mutateLink(random_scale)
    #     elif structural_mutation == 2:
    #         self.mutateNode()


    def mutate(self, weight_mutation_rate = 0.1, weight_shift_rate = 0.8, mutate_node_rate = 0.03, mutate_link_rate = 0.05, shift_scale = 0.1, random_scale = 1):
        for connection in self.connections:
            if random.random() < weight_mutation_rate:
                weight_mutation = random.choices([0, 1], weights=[weight_shift_rate, 1 - weight_shift_rate], k = 1)[0]
                if weight_mutation == 0:
                    self.mutateWeightShift(connection, shift_scale)
                elif weight_mutation == 1:
                    self.mutateWeightRandom(connection, random_scale)
        
        for node in self.nodes:
            if random.random() < weight_mutation_rate:
                bias_mutation = random.choices([1, 2], weights=[weight_shift_rate, 1 - weight_shift_rate], k = 1)[0]
                if bias_mutation == 1:
                    self.mutateBiasShift(node, shift_scale)
                elif bias_mutation == 2:
                    self.mutateBiasRandom(node, random_scale)

        if random.random() < mutate_node_rate:
            self.mutateNode()

        if random.random() < mutate_link_rate:
            self.mutateLink(random_scale)


    def mutateLink(self, random_scale = 1):
        if len(self.nodes) <= 0:
            return

        for i in range(100):
            node_in = random.choice(self.nodes)
            node_out = random.choice(self.nodes)

            if node_in.layer == node_out.layer:
                continue

            new_connection = None
            if node_in.layer < node_out.layer:
                new_connection = ConnectionGene(-1, node_in, node_out)
            else:
                new_connection = ConnectionGene(-1, node_out, node_in)

            if new_connection in self.connections:
                continue

            new_connection = self.neat.newConnection(new_connection.IN_node, new_connection.OUT_node)
            new_connection.weight = random.uniform(-random_scale, random_scale)

            self.connections.append(new_connection)

            self.nodes.sort(key=lambda x: x.innovation_number)
            self.connections.sort(key=lambda x: x.innovation_number)

            break

        
    def mutateNode(self):
        if len(self.connections) <= 0:
            return

        for i in range(100):
            connection_to_break = random.choice(self.connections)

            node_in = connection_to_break.IN_node
            node_out = connection_to_break.OUT_node

            replace_index = self.neat.getReplaceIndex(node_in, node_out)

            new_node = None

            if replace_index == -1:
                new_node = self.neat.createNewNode()
                new_node.layer = (node_in.layer + node_out.layer) / 2
                
                self.neat.setReplaceIndex(node_in, node_out, new_node.innovation_number)
            else:
                new_node = self.neat.getNode(replace_index)

            if new_node in self.nodes:
                continue

            connection_in = self.neat.newConnection(node_in, new_node)
            connection_out = self.neat.newConnection(new_node, node_out)

            connection_in.weight = 1
            connection_out.weight = connection_to_break.weight

            connection_in.enabled = True
            connection_out.enabled = connection_to_break.enabled

            connection_to_break.enabled = False
            #self.connections.pop(self.connections.index(connection_to_break))
            self.connections.append(connection_in)
            self.connections.append(connection_out)

            self.nodes.append(new_node)

            self.nodes.sort(key=lambda x: x.innovation_number)
            self.connections.sort(key=lambda x: x.innovation_number)

            break


    def mutateWeightShift(self, connection, shift_scale = 0.1):
        connection.weight += random.uniform(-shift_scale, shift_scale)


    def mutateWeightRandom(self, connection, random_scale = 1):
        connection.weight = random.uniform(-random_scale, random_scale)

    
    def mutateBiasShift(self, node, shift_scale = 0.1):
        node.bias += random.uniform(-shift_scale, shift_scale)


    def mutateBiasRandom(self, node, random_scale = 1):
        node.bias += random.uniform(-random_scale, random_scale)


    def mutateLinkToggle(self, connection):
        connection.enabled = not connection.enabled


#--------------------------------------------------------- Calculate ---------------------------------------------------------


    def calculate(self, input_list):
        #if self.calculator == None:
        self.calculator = Calculator(self)

        return self.calculator.calculate(input_list)