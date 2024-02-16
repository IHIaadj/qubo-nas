# Quantum Annealing-Based Neural Architecture Search

This repository contains the implementation of a Quantum Annealing-based Neural Architecture Search (NAS) using a Graph Neural Network (GNN) supernetwork. The project leverages the concept of Quadratic Unconstrained Binary Optimization (QUBO) to efficiently navigate the architecture search space.

## Overview

The project consists of two main components:
- `GNNSupernetwork`: A graph neural network-based supernetwork that represents a vast space of potential architectures.
- `QUBOSearch`: A class that integrates the supernetwork with a QUBO problem formulation, solving it via simulated quantum annealing to find the optimal architecture.

## Usage

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-github-username/qubo-nas.git
cd qubo-nas
python test.py
```
