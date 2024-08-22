from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class BaseQUBO:
    """Base class for defining a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    This abstract class serves as a template for QUBO problem definitions. It provides
    properties for accessing the QUBO matrix, solutions, and various lists of bitstrings
    and costs that result from solving the QUBO problem.

    Attributes:
        matrix (np.ndarray): The symmetric matrix representing the QUBO problem.
    """

    @property
    @abstractmethod
    def n_variables(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Symmetric matrix representing a QUBO problem.

        This abstract method must be implemented by subclasses to define the QUBO matrix.
        """
        raise NotImplementedError

    @property
    def solution_bitstring(self) -> np.ndarray:
        """Optimal solution bitstring for the QUBO problem.

        Returns:
            np.ndarray: The bitstring that minimizes the QUBO cost function.
        """
        return self.brute_force_solver[0]

    @property
    def solution_cost(self) -> float:
        """Cost associated with the optimal solution bitstring.

        Returns:
            float: The minimum cost value corresponding to the optimal bitstring.
        """
        return self.brute_force_solver[1]

    @property
    def degenerate_solution_bitstrings(self) -> List[np.ndarray]:
        """List of degenerate solution bitstrings.

        These are bitstrings that produce the same minimum cost as the optimal solution.

        Returns:
            List[np.ndarray]: A list of bitstrings with the same minimum cost.
        """
        return self.brute_force_solver[2]

    @property
    def binaries_list(self) -> List[np.ndarray]:
        """List of all evaluated bitstrings.

        Returns:
            List[np.ndarray]: A list of all possible bitstrings evaluated in the QUBO problem.
        """
        return self.brute_force_solver[3]

    @property
    def costs_list(self) -> List[float]:
        """List of costs corresponding to all evaluated bitstrings.

        Returns:
            List[float]: A list of costs corresponding to each bitstring evaluated.
        """
        return self.brute_force_solver[4]

    @cached_property
    def brute_force_solver(
        self, tol: float = 1.0e-10
    ) -> Tuple[np.ndarray, float, List[np.ndarray], List[np.ndarray], List[float]]:
        """Solves a QUBO problem using brute force search.

        This function performs an exhaustive search over all possible binary solutions to a QUBO
        problem to find the solution with the minimum cost. It evaluates each binary solution
        and returns the best solution along with its cost. It also returns all binary solutions
        and their corresponding energies.

        Args:
            tol (float, optional): The tolerance for considering two costs as equal.
            Defaults to 1.0e-10.

        Returns:
            Tuple[np.ndarray, float, List[np.ndarray], List[np.ndarray], List[float]]:
                - best_solution (np.ndarray): The binary solution with the minimum cost.
                - best_cost (float): The minimum cost value obtained.
                - degenerate_solution_bitstrings (List[np.ndarray]): A list of bitstrings with the
                  same minimum cost.
                - all_binaries (List[np.ndarray]): A list of all binary solutions evaluated.
                - all_costs (List[float]): A list of energies corresponding to all binary solutions.
        """
        solution_bitstring = None
        solution_cost = float("inf")
        degenerate_solution_bitstrings = []
        all_binaries = []
        all_costs = []
        for i in range(2**self.n_variables):
            bitstring = np.array(list(bin(i)[2:].zfill(self.n_variables)), dtype=int)
            cost = float(np.dot(bitstring, np.dot(self.matrix, bitstring)))
            all_binaries.append(bitstring)
            all_costs.append(cost)
            if cost - solution_cost < -tol:
                solution_cost = cost
                solution_bitstring = bitstring.copy()
                degenerate_solution_bitstrings = [solution_bitstring]
            elif abs(cost - solution_cost) < tol:
                degenerate_solution_bitstrings.append(bitstring.copy())
        return (
            solution_bitstring,
            solution_cost,
            degenerate_solution_bitstrings,
            all_binaries,
            all_costs,
        )

    def __post_init__(self) -> None:
        """Post-initialization hook for BaseQUBO.

        Can be used by subclasses to perform additional setup after the instance is created.
        """
        pass


@dataclass
class QUBO(BaseQUBO):
    _matrix: np.ndarray = field(default_factory=lambda: np.array([[-5, -2], [-2, 6]]))
    _n_variables: int = 2

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, value: np.ndarray) -> np.ndarray:
        self._matrix = value
        self._n_variables = len(value)
        return self._matrix

    @property
    def n_variables(self) -> int:
        return self._n_variables

    @n_variables.setter
    def n_variables(self, value: Any) -> None:
        raise AttributeError("This property is read-only and cannot be modified.")

    def __post_init__(self) -> None:
        return super().__post_init__()


@dataclass
class RandomQUBO(BaseQUBO):
    """Random QUBO problem generator.

    This class generates a QUBO problem with a randomly generated symmetric matrix. The random
    seed and number of variables can be specified to ensure reproducibility.

    Attributes:
        seed (Optional[int]): The seed for the random number generator to ensure reproducibility.
        n_variables (int): The number of binary variables (size of the QUBO matrix).
    """

    seed: Optional[int] = None

    @property
    def n_variables(self) -> int:
        return 2

    @n_variables.setter
    def n_variables(self, value: Any) -> None:
        raise AttributeError("This property is read-only and cannot be modified.")

    @cached_property
    def matrix(self) -> np.ndarray:
        """Generates a QUBO (Quadratic Unconstrained Binary Optimization) matrix.

        This function creates a symmetric matrix, based on a random seed and the specified number of
        binary variables.

        Returns:
            np.ndarray: A symmetric QUBO matrix of shape (n_variables, n_variables).
        """
        n_variables: int = 2
        if self.seed == 0:
            sym_matrix = np.array([[-5, -2], [-2, 6]])
        else:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)
            rnd_matrix = np.random.rand(n_variables, n_variables) - np.random.rand(
                n_variables, n_variables
            )
            sym_matrix = (rnd_matrix + rnd_matrix.T) / 2
        sym_matrix.flags.writeable = False  # Make the array immutable
        return sym_matrix

    def __post_init__(self) -> None:
        """Post-initialization hook for RandomQUBO.

        Ensures that the base class post-initialization is called.
        """
        super().__post_init__()
