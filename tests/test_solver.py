from __future__ import annotations

import numpy as np
import pytest
import torch
from qadence import QuantumCircuit

from smoqubo.solver import BaseQAOA


@pytest.fixture
def setup_base_qaoa() -> BaseQAOA:
    qubo_matrix = np.array([[-5, -2], [-2, 6]])
    n_layers = 2
    return BaseQAOA(qubo_matrix, n_layers)


def test_qubo_matrix_to_ising_matrix(setup_base_qaoa: BaseQAOA) -> None:
    const, diagonal, offdiag = setup_base_qaoa.qubo_matrix_to_ising_matrix()

    expected_const = -0.5
    expected_diagonal = np.array([3.5, -2.0])
    expected_offdiag = np.array([[0.0, -1.0], [0.0, 0.0]])

    assert np.isclose(const, expected_const), "Constant term is incorrect."
    assert np.allclose(diagonal, expected_diagonal), "Diagonal terms are incorrect."
    assert np.allclose(offdiag, expected_offdiag), "Off-diagonal terms are incorrect."


def test_cost_hamiltonian_to_circuit(setup_base_qaoa: BaseQAOA) -> None:
    circuit = setup_base_qaoa.cost_hamiltonian_to_circuit()

    # Check the type and structure of the circuit
    assert isinstance(
        circuit, QuantumCircuit
    ), "Circuit should be a QuantumCircuit instance."
    assert (
        circuit.n_qubits == setup_base_qaoa.n_qubits
    ), "Number of qubits in the circuit is incorrect."


def test_train_method(setup_base_qaoa: BaseQAOA) -> None:
    n_epochs = 10
    n_logs_loss_history = 5
    lr = 0.1
    final_loss, loss_history = setup_base_qaoa.train(n_epochs, n_logs_loss_history, lr)
    assert isinstance(final_loss, torch.Tensor), "Final loss should be a tensor."
    assert isinstance(loss_history, list), "Loss history should be a list."
    assert loss_history[-1] < loss_history[0], "Training did not reduce the loss."


def test_loss_function(setup_base_qaoa: BaseQAOA) -> None:
    loss = setup_base_qaoa.model.expectation()[0]
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."
