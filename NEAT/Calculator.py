class Calculator:

    def __init__(self, genome):
        self.input_list = []
        self.hidden_list = []
        self.output_list = []

        nodes = genome.nodes.copy()
        connections = genome.connections.copy()

        for node in nodes:
            if node.layer <= 0:
                self.input_list.append(node)
            elif node.layer >= 1:
                self.output_list.append(node)
            else:
                self.hidden_list.append(node)

        self.hidden_list.sort(key=lambda x: (x.layer, x.innovation_number))

        for connection in connections:
            if connection not in connection.OUT_node.connections:
                connection.OUT_node.connections.append(connection)

    
    def calculate(self, input_list):
        if len(input_list) != len(self.input_list):
            return None

        for i in range(len(self.input_list)):
            self.input_list[i].output = input_list[i]

        for node in self.hidden_list:
            node.calculate()

        output = []
        for node in self.output_list:
            output.append(node.calculate())

        return output