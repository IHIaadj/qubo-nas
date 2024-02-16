import logging
from qubo.search import QUBOSearch

def setup_logging():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_qubo_search(dataset, num_layers, num_operations, num_graph_layers, alpha, beta, num_iterations):
    # Initialize QUBOSearch object
    qubo_search = QUBOSearch(num_layers, num_operations, num_graph_layers, alpha, beta)

    # Run quantum annealing multiple times
    for i in range(5):
        logging.info(f"Running quantum annealing iteration: {i+1}")
        optimal_architecture = qubo_search.quantum_annealing(dataset, num_iterations)
        logging.info(f"Iteration {i+1}: Optimal Architecture - {optimal_architecture}")

if __name__ == "__main__":
    setup_logging()

    # Define your parameters
    dataset = 'CIFAR-10' 
    num_layers = 10
    num_operations = 5 
    num_graph_layers = 3 
    alpha = 0.8
    beta = 0.1
    num_iterations = 1000 

    test_qubo_search(dataset, num_layers, num_operations, num_graph_layers, alpha, beta, num_iterations)
