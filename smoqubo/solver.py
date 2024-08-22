from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from qadence import (
    RX,
    HamEvo,
    I,
    Parameter,
    QuantumCircuit,
    QuantumModel,
    Tensor,
    Z,
    Zero,
    chain,
    kron,
    tag,
)
from qadence.backend import ConvertedObservable
from qadence.constructors.hamiltonians import interaction_zz


@dataclass
class BaseQAOA:
    """Base class for constructing and training a QAOA (Quantum Approximate Optimization Algorithm)
       model.

    This class provides methods to convert a QUBO matrix to its Ising representation,
    construct the corresponding cost Hamiltonian, and build the quantum circuit for QAOA.
    Additionally, it offers a method to train the QAOA model using gradient-based optimization.

    Attributes:
        qubo_matrix (np.ndarray): The QUBO matrix representing the optimization problem.
        n_layers (int): The number of layers in the QAOA circuit.
        n_qubits (int): The number of qubits in the quantum circuit.
        ising_matrix (Tuple[np.ndarray, np.ndarray, np.ndarray]): The Ising model representation of
        the QUBO matrix.
        cost_hamiltonian (np.ndarray): The Ising Hamiltonian derived from the Ising matrix.
        circuit (QuantumCircuit): The quantum circuit representing the QAOA model.
        model (QuantumModel): The quantum model used for optimization.

    Methods:
        train: Trains the QAOA model.
        cost_hamiltonian_to_circuit: Generates the QAOA circuit blocks.
        ising_matrix_to_ising_hamiltonian: Converts an Ising matrix into an Ising Hamiltonian.
        qubo_matrix_to_ising_matrix: Converts a QUBO matrix to an Ising model matrix representation.
    """

    qubo_matrix: np.ndarray
    n_layers: int

    def __post_init__(self) -> None:
        """Initializes additional attributes after object creation.

        Converts the QUBO matrix to its Ising representation, constructs the cost Hamiltonian,
        and builds the corresponding quantum circuit for the QAOA model.
        """
        self.n_qubits: int = len(self.qubo_matrix)
        self.ising_matrix = self.qubo_matrix_to_ising_matrix()
        self.cost_hamiltonian = self.ising_matrix_to_ising_hamiltonian()
        self.circuit = self.cost_hamiltonian_to_circuit()
        self.model = QuantumModel(self.circuit, observable=self.cost_hamiltonian)

    def train(
        self,
        n_epochs: int,
        n_logs_loss_history: int,
        lr: float = 0.05,
    ) -> Tuple[Tensor, List[float]]:
        """Trains a QAOA (Quantum Approximate Optimization Algorithm) model.

        This method trains a QAOA model for a specified number of epochs using the Adam optimizer.
        It records the loss at specified intervals and prints the cost at regular checkpoints.

        Args:
            n_epochs (int): Total number of epochs to train the model.
            n_logs_loss_history (int): Number of epochs between logging the loss history.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.05.

        Returns:
            Tuple[Tensor, List[float]]:
                - Tensor: The final loss value after training.
                - List[float]: A list containing the loss values at the specified logging intervals.

        Raises:
            ValueError: If `n_logs_loss_history` is zero or negative, or if `n_epochs` is not a
            positive integer.
        """
        self.model.reset_vparams(torch.rand(self.model.num_vparams))
        initial_loss = self.model.expectation()[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print(f"Initial loss: {initial_loss}\n")
        n_logs_loss_history = min(n_epochs, 50)

        if n_logs_loss_history <= 0:
            raise ValueError("`n_logs_loss_history` must be a positive integer.")
        if n_epochs <= 0:
            raise ValueError("`n_epochs` must be a positive integer.")

        loss_history = []

        for i in range(n_epochs):
            optimizer.zero_grad()
            loss = self.model.expectation()[0]
            loss.backward()
            optimizer.step()

            if (i + 1) % (n_epochs // 10) == 0:
                print(f"Cost at iteration {i+1}: {loss.item()}")

            if (i + 1) % (n_epochs // n_logs_loss_history) == 0:
                loss_history.append(loss.item())

        return loss, loss_history

    def cost_hamiltonian_to_circuit(self) -> QuantumCircuit:
        """Generates QAOA (Quantum Approximate Optimization Algorithm) blocks.

        This method constructs QAOA blocks consisting of alternating cost and mixing layers.
        Each cost layer is derived from the given cost Hamiltonian and is followed by a mixing layer
        of single-qubit rotations. The method creates these layers for the specified number of
        layers and returns the combined blocks.

        Returns:
            QuantumCircuit: A quantum circuit representing the combined QAOA blocks.

        Raises:
            ValueError: If the number of qubits is not a positive integer or if the cost Hamiltonian
            does not match the expected dimensions.

        See Also:
            `HamEvo.digital_decomposition` for details on how cost layers are decomposed.
            `RX` and `Parameter` for details on single-qubit rotations and parameters.
        """
        layers = []
        for layer in range(self.n_layers):
            # Cost layer with digital decomposition
            cost_layer = HamEvo(
                self.cost_hamiltonian, f"g{layer}"
            ).digital_decomposition()
            cost_layer = tag(cost_layer, "cost")

            # Mixing layer with single qubit rotations
            beta = Parameter(f"b{layer}")
            mixing_layer = kron(RX(i, beta) for i in range(self.n_qubits))
            mixing_layer = tag(mixing_layer, "Mixing operator")

            # Putting all together in a single ChainBlock
            layers.append(chain(cost_layer, mixing_layer))

        blocks = chain(*layers)

        circuit = QuantumCircuit(self.n_qubits, blocks)
        return circuit

    def ising_matrix_to_ising_hamiltonian(
        self,
    ) -> list[ConvertedObservable] | ConvertedObservable:
        """Converts an Ising matrix into an Ising Hamiltonian.

        This function takes an Ising matrix representation, which consists of a constant term,
        diagonal elements (Z terms), and off-diagonal elements (ZZ interactions), and converts
        it into an Ising Hamiltonian. The Ising Hamiltonian is constructed by summing the ZZ
        interaction terms between qubits and applying local Z terms to individual qubits.

        Args:
            ising_matrix (np.ndarray): A list-like structure containing three elements:
                - const (float): A scalar constant term.
                - diagonal (np.ndarray): A 1D array representing the linear coefficients (Z terms)
                for each qubit.
                - offdiag (np.ndarray): A 2D array representing the quadratic coefficients
                (ZZ interactions) between qubits.
            self.n_qubits (int): The number of qubits in the system.

        Returns:
            list[ConvertedObservable] | ConvertedObservable: The resulting Ising Hamiltonian as a
            matrix.

        Raises:
            ValueError: If the `ising_matrix` does not contain exactly three elements.
        """
        const, diagonal, offdiag = self.ising_matrix

        ising_hamiltonian = Zero()

        # Add ZZ interaction terms based on off-diagonal elements
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                w = offdiag[i, j]
                if w != 0:
                    ising_hamiltonian += w * interaction_zz(i, j)

        # Add Z terms based on diagonal elements
        for i in range(self.n_qubits):
            w = diagonal[i]
            if w != 0:
                ising_hamiltonian += w * Z(i)

        # Add the constant term to the Hamiltonian
        ising_hamiltonian += I(i) * const

        return ising_hamiltonian

    def qubo_matrix_to_ising_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts a QUBO matrix to an Ising model matrix representation.

        This function transforms a symmetric QUBO matrix into its equivalent Ising model
        representation, which includes a constant term, a diagonal vector (representing
        linear coefficients), and an upper triangular matrix (representing quadratic coefficients).

        Returns:
            Tuple containing:
                - const (np.ndarray): A scalar constant term.
                - diagonal (np.ndarray): A 1D array of linear coefficients (Ising matrix diagonal).
                - offdiag (np.ndarray): An upper triangular matrix of quadratic coefficients
                (off-diagonal terms of the Ising matrix).

        Raises:
            AssertionError: If the input QUBO matrix is not symmetric.
        """
        assert np.array_equal(
            self.qubo_matrix, self.qubo_matrix.T
        ), "Matrix is not symmetric"
        const = np.sum(np.triu(self.qubo_matrix)) / 2
        diagonal = -np.sum(self.qubo_matrix, axis=0) / 2
        offdiag = np.triu(self.qubo_matrix, 1) / 2
        return const, diagonal, offdiag


def loss_function(model: QuantumModel) -> Tensor:
    """Computes the loss for a given quantum model.

    This function calculates the loss value for a quantum model by obtaining the expectation
    value from the model's `expectation` method. The expectation value is typically used as
    a measure of the model's performance or energy, depending on the quantum algorithm or
    problem being solved.

    Args:
        model (QuantumModel): The quantum model for which the loss is to be computed.

    Returns:
        Tensor: The loss value, represented as a tensor. This is typically a scalar tensor
        representing the expectation value of the quantum model.

    Raises:
        ValueError: If the `expectation` method of the model does not return the expected output.
    """
    val = model.expectation()[0]
    return val
