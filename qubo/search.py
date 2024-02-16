import numpy as np
from ..search_space.train import train 
from ..search_space.gnn_supernetwork import GNNSupernetwork 

class QUBOSearch:
    def __init__(self, num_layers, num_operations, num_graph_layers, alpha, beta):
        self.supernetwork = GNNSupernetwork(num_layers, num_operations, num_graph_layers)
        self.alpha = alpha
        self.beta = beta



    def objective_function(self, architecture_vector, dataset):
        """
        Compute the objective function for the QUBO problem.

        :param architecture_vector: Binary vector representing the architecture.
        :param dataset: Dataset used for evaluating accuracy.
        :return: Value of the objective function.
        """
        accuracy = self.supernetwork.compute_validation_accuracy(dataset, architecture_vector)
        complexity = self.compute_complexity(architecture_vector)
        return -self.alpha * accuracy + self.beta * complexity

    def compute_complexity(self, architecture_vector):
        """
        Compute the complexity of the architecture, defined as the number of parameters.

        :param architecture_vector: Binary vector representing the architecture.
        :return: Complexity of the architecture.
        """
        network = self.supernetwork.construct_network(architecture_vector)
        num_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
        return num_parameters

    def quantum_annealing(self, dataset, num_iterations=1000):
        """
        Simulate the quantum annealing process to find the optimal architecture.

        :param dataset: Dataset used for evaluating accuracy.
        :param num_iterations: Number of iterations for the simulated annealing.
        :return: Optimal architecture vector.
        """

        train(dataset, self.supernetwork)
        best_architecture = None
        best_objective_value = float('inf')

        for _ in range(num_iterations):
            architecture_vector = np.random.randint(2, size=self.supernetwork.num_layers * self.supernetwork.num_operations)

            # Compute the objective function value for this architecture
            objective_value = self.objective_function(architecture_vector, dataset)

            # Update the best architecture if this is the best objective value so far
            if objective_value < best_objective_value:
                best_architecture = architecture_vector
                best_objective_value = objective_value

        return best_architecture

