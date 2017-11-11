import numpy as np
import random
import statistics

class BasicPopiNoBias:
    def __init__(self):
        self.input_size = None
        self.output_size = None

        self.hidden_layer_sizes = None
        self.layers = None

        self.fitness = None

    def logistic(self, x):
        e = 2.71828
        return 1/(1+e**(-x))

    def initialize_network(self):
        # randomly initializes the data structures for the neural network. self.hidden_layer_sizes should be set first
        if self.hidden_layer_sizes is None:
            print('ERROR: attempting to initialize neural network in popi without defined hidden_layer_sizes')
            return
        elif self.input_size is None:
            print('ERROR: attempting to initialize neural network in popi without defined input size')
            return
        elif self.output_size is None:
            print('ERROR: attempting to initialize neural network in popi without defined output size')
            return

        self.layers = list()
        next_layer_input_size = self.input_size
        for l in self.hidden_layer_sizes:
            new_layer = np.matrix(np.random.rand(next_layer_input_size, l))
            self.layers.append(new_layer)
            next_layer_input_size = l

        last_layer = np.matrix(np.random.rand(next_layer_input_size, self.output_size))
        self.layers.append(last_layer)

    def recombine(self, other):
        # returns a new popi by recombining the network weight values between self and other
        new_child = BasicPopiNoBias()

        # copy member values over
        new_child.input_size = self.input_size
        new_child.output_size = self.output_size
        new_child.hidden_layer_sizes = self.hidden_layer_sizes

        # recombine the network layers one by one
        num_layers = len(self.layers)
        new_child.layers = []
        for i in range(num_layers):
            layer_shape = self.layers[i].shape
            recombination_mask = np.random.rand(layer_shape[0], layer_shape[1]) < 0.5
            recombination_mask2 = recombination_mask == False
            new_layer = np.multiply(self.layers[i],  recombination_mask) + np.multiply(other.layers[i], recombination_mask2)
            new_child.layers.append(new_layer)

        return new_child

    def get_output(self, input):
        next_layer_input = input
        output = next_layer_input

        for l in self.layers:
            z = next_layer_input * l
            #output = scipy.stats.logistic.cdf(z)
            output = self.logistic(np.array(z))
            next_layer_input = output

        return output


class BasicPopi:
    def __init__(self):
        self.input_size = None
        self.output_size = None

        self.hidden_layer_sizes = None
        self.layers = None

        self.fitness = None

    def logistic(self, x):
        e = 2.71828
        return 1/(1+e**(-x))

    def initialize_network(self):
        # randomly initializes the data structures for the neural network. self.hidden_layer_sizes should be set first
        if self.hidden_layer_sizes is None:
            print('ERROR: attempting to initialize neural network in popi without defined hidden_layer_sizes')
            return
        elif self.input_size is None:
            print('ERROR: attempting to initialize neural network in popi without defined input size')
            return
        elif self.output_size is None:
            print('ERROR: attempting to initialize neural network in popi without defined output size')
            return

        self.layers = list()
        next_layer_input_size = self.input_size
        for l in self.hidden_layer_sizes:
            new_layer = np.matrix(np.random.rand(next_layer_input_size + 1, l))
            self.layers.append(new_layer)
            next_layer_input_size = l

        last_layer = np.matrix(np.random.rand(next_layer_input_size + 1, self.output_size))
        self.layers.append(last_layer)

    def recombine(self, other):
        # returns a new popi by recombining the network weight values between self and other
        new_child = BasicPopi()

        # copy member values over
        new_child.input_size = self.input_size
        new_child.output_size = self.output_size
        new_child.hidden_layer_sizes = self.hidden_layer_sizes

        # recombine the network layers one by one
        num_layers = len(self.layers)
        new_child.layers = []
        for i in range(num_layers):
            layer_shape = self.layers[i].shape
            recombination_mask = np.random.rand(layer_shape[0], layer_shape[1]) < 0.5
            recombination_mask2 = recombination_mask == False
            new_layer = np.multiply(self.layers[i],  recombination_mask) + np.multiply(other.layers[i], recombination_mask2)
            new_child.layers.append(new_layer)

        return new_child



    def get_output(self, input):
        next_layer_input = np.concatenate([[1], input])
        output = next_layer_input

        for l in self.layers:
            z = next_layer_input * l
            output = np.ndarray.flatten(self.logistic(np.array(z)))
            next_layer_input = np.concatenate([[1], output])

        return output


class NEATPopi:
    class Node:
        def __init__(self):
            self.calculated = False
            self.value = None


            self.inputs = [] # list of tuples of (input node, associated connection)
            self.function = None

            self.id = None

        @staticmethod
        def logistic(x):
            e = 2.71828
            return 1 / (1 + e ** (-x))

        def get_value(self, ancestor_ids):
            if self.calculated:
                return self.value

            # prevent loops
            elif self.id in ancestor_ids:
                return 0

            else:
                self.value = 0
                for n, c in self.inputs:
                    if c.enabled:
                        new_ancestor_ids = ancestor_ids.copy()
                        new_ancestor_ids.add(self.id)
                        self.value += n.get_value(new_ancestor_ids)
                self.value = self.logistic(self.value)
                self.calculated = True
                return self.value

    class Connection:
        def __init__(self):
            self.input_id = None
            self.output_id = None
            self.weight = None
            self.innovation = None
            self.enabled = None

    def __init__(self):
        self.input_size = None
        self.output_size = None

        self.nodes = dict()
        self.input_nodes = dict()
        self.output_nodes = dict()
        self.hidden_nodes =  dict()

        self.connections = dict()

        self.input = None
        self.output = None

        self.next_node_id = 0

    def initialize_inputs_outputs(self):

        if self.input_size is None:
            print('ERROR: attempting to initialize neural network in NEATpopi without defined input size')
            return
        elif self.output_size is None:
            print('ERROR: attempting to initialize neural network in NEATpopi without defined output size')
            return

        for i in range(self.input_size):
            new_node = self.Node()
            new_node.function = 'input'
            new_node.id = self.next_node_id
            self.next_node_id += 1

            self.input_nodes[new_node.id] = new_node
            self.nodes[new_node.id] = new_node

        for o in range(self.output_size):
            new_node = self.Node()
            new_node.function = 'output'
            new_node.id = self.next_node_id
            self.next_node_id += 1

            self.output_nodes[new_node.id] = new_node
            self.nodes[new_node.id] = new_node

    def similarity(self, other):
        c1 = 1
        c2 = 1
        c3 = 1

        weight_differences = []
        num_disjoint_genes = 0
        num_excess_genes = 0

        all_connection_ids = set(self.connections.keys()).union(set(other.connections.keys()))

        if len(self.connections) == 0:
            self_innovation = 0
        else:
            self_innovation = max(self.connections.keys())

        if len(other.connections) == 0:
            other_innovation = 0
        else:
            other_innovation = max(other.connections.keys())

        excess_threshold = min(self_innovation, other_innovation)

        for c_id in all_connection_ids:
            if c_id in self.connections.keys() and c_id in other.connections.keys():
                weight_difference = abs(self.connections[c_id].weight - other.connections[c_id].weight)
                weight_differences.append(weight_difference)
            elif c_id > excess_threshold:
                num_excess_genes += 1
            else:
                num_disjoint_genes += 1

        N = max(len(self.connections), len(other.connections))

        delta = 0
        delta += c1 * num_excess_genes / N
        delta += c2 * num_disjoint_genes / N
        if len(weight_differences) > 0:
            delta += c3 * statistics.mean(weight_differences)

        return delta


    def mutate_add_connection(self, innovation_number):
        node1, node2 = random.sample(list(self.nodes.values()), 2)
        new_connection = self.Connection()

        new_connection.weight = random.uniform(-1, 1)
        new_connection.input_id = node1.id
        new_connection.output_id = node2.id
        new_connection.innovation = innovation_number
        new_connection.enabled = True

        node2.inputs.append((node1, new_connection))

        self.connections[innovation_number] = new_connection

    def mutate_add_node(self, innovation_number):
        if len(self.connections) > 0:
            old_connection = random.choice(list(self.connections.values()))

            node1_id = old_connection.input_id
            node1 = self.nodes[node1_id]

            node3_id = old_connection.output_id
            node3 = self.nodes[node3_id]

            node2 = self.Node()
            node2.id = self.next_node_id
            self.next_node_id += 1

            new_connection1 = self.Connection
            new_connection1.weight = 1
            new_connection1.input_id = node1.id
            new_connection1.output_id = node2.id
            new_connection1.innovation = innovation_number
            new_connection1.enabled = True

            new_connection2 = self.Connection
            new_connection2.weight = old_connection.weight
            new_connection2.input_id = node2.id
            new_connection2.output_id = node3.id
            new_connection2.innovation = innovation_number + 1
            new_connection2.enabled = True

            node2.inputs.append((node1, new_connection1))
            node3.inputs.append((node2, new_connection2))

            old_connection.enabled = False

            self.connections[innovation_number] = new_connection1
            self.connections[innovation_number + 1] = new_connection2

            self.hidden_nodes[node2.id] = node2
            self.nodes[node2.id] = node2

    def mutate_perturb_weights(self, perturbation_rate):
        for c in self.connections:
            if random.random() < perturbation_rate:
                c.weight += random.triangular(-1, 1, 0)

    def recombine(self, other):
        new_child = NEATPopi()

        new_child.input_size = self.input_size
        new_child.output_size = self.output_size

        new_child.initialize_inputs_outputs()

        for c_id in self.connections.keys():
            if c_id in other.connections.keys() and random.random():
                connection_to_copy = other.connections[c_id]
            else:
                connection_to_copy = self.connections[c_id]

            new_connection = self.Connection()
            new_connection.weight = connection_to_copy.weight
            new_connection.innovation = connection_to_copy.innovation
            new_connection.enabled = connection_to_copy.enabled

            node1_id = connection_to_copy.input_id
            node2_id = connection_to_copy.output_id

            for n_id in (node1_id,  node2_id):
                if n_id not in (n.id for n in new_child.nodes.values()):
                    new_node = self.Node()
                    new_node.id = n_id

                    new_child.hidden_nodes[new_node.id] = new_node
                    new_child.nodes[new_node.id] = new_node

            node1 = new_child.nodes[node1_id]
            node2 = new_child.nodes[node2_id]
            node2.inputs.append((node1, new_connection))

            new_connection.input_id = node1_id
            new_connection.output_id = node2.id

            new_child.connections[c_id] = new_connection

        return new_child

    def get_output(self, input):
        for n in self.nodes.values():
            n.calculated = False

        for i in range(len(input)):
            self.input_nodes[i].value = input[i]
            self.input_nodes[i].calculated = True

        output = []
        for n in self.output_nodes.values():
            output.append(n.get_value(set()))

        return output

