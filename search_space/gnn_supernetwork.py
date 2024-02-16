import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data

from layer_ops import * 

class GNNSupernetwork(nn.Module):
    def __init__(self, num_layers, num_operations, num_graph_layers):
        super(GNNSupernetwork, self).__init__()
        self.num_layers = num_layers
        self.num_operations = num_operations
        self.initial_in_channels = 3
        # Define the operations for each layer
        self.operations = nn.ModuleDict({
            'BasicBlock': BasicBlock(64, 128, 1),
            'BottleneckBlock': BottleneckBlock(128, 256, 1),
            'VGGBlock': VGGBlock(256, 512, 3, 1),
            'MBConv': MBConv(32, 64, 3, 1, 6),
            'VisionAttentionBlock': VisionAttentionBlock(256, 8),
            'ZeroOps': ZeroOps()
        })

        # Define the GNN layers
        self.gnn_layers = nn.ModuleList([
            geom_nn.GraphConv(num_operations, num_operations) for _ in range(num_graph_layers)
        ])

    def interpret_operations(self, node_features):
        # Applying softmax to get probabilities for each operation
        operation_probs = torch.softmax(node_features, dim=1)
        
        # Selecting the operation with the highest probability
        selected_operations = torch.argmax(operation_probs, dim=1)

        # Convert to binary (one-hot) format
        one_hot_operations = F.one_hot(selected_operations, num_classes=self.num_operations)

        return one_hot_operations
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Process through GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        # Interpret the GNN output to select operations for each layer
        operation_matrix = self.interpret_operations(x)

        return operation_matrix
    
    def construct_network(self, architecture_vector):
        layers = []
        in_channels = self.initial_in_channels 

        for i, operation_idx in enumerate(architecture_vector):
            if operation_idx == self.operation_indices['ZeroOps']:
                continue  # Skip this layer (ZeroOps)

            operation = self.operations[operation_idx]
            if operation_idx in [self.operation_indices['BasicBlock'], self.operation_indices['BottleneckBlock'], self.operation_indices['VGGBlock'], self.operation_indices['MBConv'], self.operation_indices['VisionAttentionBlock']]:
                # For these operations, pass in_channels and calculate out_channels
                layer = operation(in_channels)
                out_channels = layer.out_channels
            elif operation_idx == self.operation_indices['Conv']:
                # For Conv, define additional parameters like kernel size
                layer = operation(in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
                out_channels = layer.out_channels
            elif operation_idx in [self.operation_indices['MaxPool'], self.operation_indices['AvgPool']]:
                # For pooling layers
                layer = operation(kernel_size=2, stride=2)

            layers.append(layer)
            in_channels = out_channels  # Update in_channels for the next layer

        return nn.Sequential(*layers)
    
    def compute_validation_accuracy(self, validation_loader, architecture_vector):
        # Construct the network based on the architecture vector
        network = self.construct_network(architecture_vector)
        
        # Evaluate the network on the validation dataset
        correct = 0
        total = 0
        network.eval()  # Set the network to evaluation mode
        with torch.no_grad():
            for inputs, labels in validation_loader:
                outputs = network(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy
    
    def compute_nb_param(self, architecture_vector):
        # Construct the network based on the architecture vector
        network = self.construct_network(architecture_vector)
        return sum(p.numel() for p in network.parameters() if p.requires_grad)

    
# TEST
# Define the number of layers, operations, and graph convolutional layers
num_layers = 10 # Define based on your architecture
num_operations = 6 # Define the number of operations 
num_graph_layers = 2 # Define the number of GNN layers

# Instantiate the supernetwork
gnn_supernetwork = GNNSupernetwork(num_layers, num_operations, num_graph_layers)

