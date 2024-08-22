from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from qadence.draw import savefig

from smoqubo.generator import RandomQUBO
from smoqubo.solver import BaseQAOA

# Solve the simple QUBO matrix M presented in the documentation
# |-5 -2|
# |-2  6|

# Define a qubo object: seed=0 sets M as qubo matrix
qubo = RandomQUBO(0)
print(f"\nQUBO matrix\n{qubo.matrix}\n")

# Obtain the solution with a brute force solver
print("Brute force solution")
print(f"Solution bitstring: {qubo.solution_bitstring}")
print(f"Solution cost: {qubo.solution_cost}")
print(f"Costs list: {qubo.costs_list}\n")
if len(qubo.degenerate_solution_bitstrings) > 1:
    print(f"Degeneracy degree: {len(qubo.degenerate_solution_bitstrings)}")
    print(f"Degenerate solution bitstrings: {qubo.degenerate_solution_bitstrings}\n")

# Define a qaoa object
n_layers = 4
qaoa = BaseQAOA(qubo.matrix, n_layers)

# Some information
print(
    f"\nIsing Matrix:\n{qubo.matrix}",
)
print(f"Constant: {qaoa.ising_matrix[0]}")
print(f"Diagonal: {qaoa.ising_matrix[1]}")
print(f"Off Diagonal\n {qaoa.ising_matrix[2]}\n")

# Save an illustration of the qaoa circuit
savefig(qaoa.circuit, "./img/qaoa_circuit.png")

# Train the circuit
n_epochs = 100
n_logs_loss_history = min(n_epochs, 50)
lr = 0.05
qaoa.train(n_epochs, n_logs_loss_history, lr)
print(f"Exact solution cost: {qubo.solution_cost}")

# After training, test the qaoa.model
n_variables = 2
n_qubits = n_variables
n_shots = 3_600

wf = qaoa.model.run()
xs = qaoa.model.sample(n_shots=n_shots)[0]
ex = qaoa.model.expectation()
samples = qaoa.model.sample(n_shots=n_shots)[0]
most_frequent = max(samples, key=samples.get)
print(f"\nQAOA solution bitstring: [{' '.join(most_frequent)}]")
print(f"Exact solution bitstring: {qubo.solution_bitstring}")

# Extract keys and values
labels = list(samples.keys())
values = np.array(list(samples.values())) / n_shots

# Plotting the bar chart
plt.grid(True, linestyle="-.", zorder=0)
plt.bar(labels, values, zorder=3)
# Adding labels and title
plt.xlabel("Bitstrings")
plt.ylabel("Frequency")
plt.title("Sampling of QAOA output, bar chart")
plt.xticks(rotation="vertical")

# Store the plot
plt.savefig("./img/sampling.png")
