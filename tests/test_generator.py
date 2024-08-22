from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from smoqubo.generator import QUBO, BaseQUBO, RandomQUBO


class TestBaseQUBO(BaseQUBO):
    """A simple implementation of BaseQUBO for testing purposes."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix
        self._n_variables = len(matrix)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def n_variables(self) -> int:
        return self._n_variables

    @n_variables.setter
    def n_variables(self, value: Any) -> None:
        raise AttributeError("This property is read-only and cannot be modified.")


@pytest.fixture
def setup_qubo() -> TestBaseQUBO:
    """Fixture for setting up a simple QUBO matrix."""
    matrix = np.array([[1, -2], [-2, 4]])
    return TestBaseQUBO(matrix)


@pytest.fixture
def setup_degenerate_qubo() -> QUBO:
    """Fixture for setting up a QUBO matrix with degenerate solutions."""
    matrix = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    qubo = QUBO()
    qubo.matrix = matrix
    return qubo


@pytest.fixture
def setup_random_qubo() -> RandomQUBO:
    """Fixture for setting up a RandomQUBO instance."""
    return RandomQUBO(seed=42)


def test_solution_bitstring(setup_qubo: TestBaseQUBO) -> None:
    """Test for the solution bitstring property."""
    bitstring = setup_qubo.solution_bitstring
    expected_bitstring = np.array([0, 0])
    assert np.array_equal(
        bitstring, expected_bitstring
    ), "Solution bitstring is incorrect."


def test_solution_cost(setup_qubo: TestBaseQUBO) -> None:
    """Test for the solution cost property."""
    cost = setup_qubo.solution_cost
    expected_cost = 0.0  # Expected cost from the matrix and solution bitstring
    assert cost == expected_cost, "Solution cost is incorrect."


def test_degenerate_solution_bitstrings(setup_degenerate_qubo: QUBO) -> None:
    """Test for the degenerate solution bitstrings property."""
    degenerate_bitstrings = setup_degenerate_qubo.degenerate_solution_bitstrings
    expected_bitstrings = [np.array([0, 0, 0]), np.array([1, 1, 1])]
    assert len(degenerate_bitstrings) == len(expected_bitstrings)
    assert all(
        np.array_equal(d, e) for d, e in zip(degenerate_bitstrings, expected_bitstrings)
    ), "Degenerate solution bitstrings are incorrect."


def test_binaries_list(setup_qubo: TestBaseQUBO) -> None:
    """Test for the binaries list property."""
    binaries = setup_qubo.binaries_list
    expected_binaries = [
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([1, 1]),
    ]
    assert len(binaries) == len(expected_binaries)
    assert all(
        np.array_equal(b, e) for b, e in zip(binaries, expected_binaries)
    ), "Binaries list is incorrect."


def test_costs_list(setup_qubo: TestBaseQUBO) -> None:
    """Test for the costs list property."""
    costs = setup_qubo.costs_list
    expected_costs = [0.0, 4.0, 1.0, 1.0]
    assert costs == expected_costs, "Costs list is incorrect."


def test_random_qubo_matrix_generation(setup_random_qubo: RandomQUBO) -> None:
    """Test for random QUBO matrix generation."""
    matrix = setup_random_qubo.matrix
    expected_matrix_shape = (2, 2)
    assert (
        matrix.shape == expected_matrix_shape
    ), "Generated QUBO matrix has incorrect shape."
    assert np.allclose(matrix, matrix.T), "Generated QUBO matrix is not symmetric."


def test_random_qubo_deterministic_generation() -> None:
    """Test for deterministic random QUBO generation with a fixed seed."""
    qubo_1 = RandomQUBO(seed=42)
    qubo_2 = RandomQUBO(seed=42)
    assert np.array_equal(
        qubo_1.matrix, qubo_2.matrix
    ), "Matrices generated with the same seed should be identical."


def test_random_qubo_custom_seed() -> None:
    """Test for custom seed matrix generation in RandomQUBO."""
    qubo = RandomQUBO(seed=0)
    expected_matrix = np.array([[-5, -2], [-2, 6]])
    assert np.array_equal(
        qubo.matrix, expected_matrix
    ), "Custom seed matrix generation incorrect"
