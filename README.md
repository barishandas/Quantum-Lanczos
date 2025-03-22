# Quantum Lanczos Algorithm for H₂ Molecular Simulation

This repository contains an implementation of the Quantum Lanczos algorithm for calculating the ground state energy of the H₂ molecule at various bond distances. The code demonstrates how quantum computing techniques can be applied to computational chemistry problems, with potential extensions to carbon capture materials research.

## Overview

The `lanczosh2.py` script implements the Quantum Lanczos algorithm, which uses a Krylov subspace approach to approximate the ground state of quantum systems. For the H₂ molecule, the script:

1. Creates Hamiltonians for different H-H bond distances
2. Builds a Krylov subspace through quantum circuit operations
3. Constructs and diagonalizes a tridiagonal matrix representation
4. Calculates ground state energies and plots the potential energy curve

## Requirements

- Python 3.8+
- NumPy
- Qiskit
- Matplotlib
- SciPy

## Usage

```bash
python lanczosh2.py
```

This will run a bond distance scan from 0.5 to 2.0 Angstroms, calculate energies at each point, and generate a potential energy curve plot showing the equilibrium bond distance.

## Main Components

- `QuantumLanczos`: Main class implementing the algorithm
- `create_h2_hamiltonian()`: Function to create H₂ molecular Hamiltonians
- `run_distance_scan()`: Function to calculate energies across multiple bond distances
- `plot_energy_curve()`: Function to visualize the potential energy surface

## Extension to CO₂ Capture

This implementation serves as a foundation for extending quantum computational methods to study carbon capture materials:

- The same approach can be applied to calculate binding energies of CO₂ in metal-organic frameworks
- More complex Hamiltonians can be constructed to include metal centers and CO₂ interactions
- The quantum advantage may become significant for larger systems where classical methods struggle

## Output

The script generates a plot file (`h2_quantum_lanczos.png`) showing the potential energy curve with the equilibrium distance marked.

## Future Work

- Extend to larger molecular systems
- Implement more efficient Trotterization schemes
- Add support for excited states calculations
- Incorporate noise models to simulate real quantum hardware effects
