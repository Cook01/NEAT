from NEAT.Gene import Gene

# Genes that represent a connection between two nodes
class ConnectionGene(Gene):

    def __init__(self, innovation_number, IN_node, OUT_node):
        super().__init__(innovation_number)

        self.IN_node = IN_node
        self.OUT_node = OUT_node

        self.weight = 1
        self.enabled = True

        self.replace_index = -1



    def __eq__(self, other):
       if type(self) != type(other): return False
    
       return (self.IN_node == other.IN_node) and (self.OUT_node == other.OUT_node)



    def __repr__(self):
        return str(self.innovation_number) + " || " + str(self.IN_node) + " -> " + str(self.OUT_node) + " || Weight : " + str(self.weight) + " || Enabled : " + str(self.enabled)
    
    def __str__(self):
        return str(self.innovation_number) + " || " + str(self.IN_node) + " -> " + str(self.OUT_node) + " || Weight : " + str(self.weight) + " || Enabled : " + str(self.enabled)


#--------------------------------------------------------- Utils ---------------------------------------------------------


    # Return a copy of the Connection
    def copy(self):
        new_connection = ConnectionGene(self.innovation_number, self.IN_node, self.OUT_node)
        new_connection.weight = self.weight
        new_connection.enabled = self.enabled

        return new_connection