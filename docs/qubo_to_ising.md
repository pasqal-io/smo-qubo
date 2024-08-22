# QUBO matrix to Ising Hamiltonian

Here we discuss how the matrix formulation of a QUBO problem maps to a Ising Hamiltonian.

## Example

To explain how to solve the problem with QAOA, let us use a simple example.
Consider minimizing the following 2x2 QUBO objective function:

$$
\left(\begin{array}{c}
x_1 \quad x_2
\end{array}\right)
\left(\begin{array}{cc}
-5 & -2\\
-2 & 6
\end{array}\right)
\left(\begin{array}{c}
x_1 \\
x_2
\end{array}\right)
=  - 4 x_1 x_2 - 5 x_1^2 + 6 x_2^2
$$

The function is minimized by $(x_1, \ x_2)=(1,0)$, and the corresponding objective function value is -5.

The weights are: $[ \ w_{12}=-4, \ w_{11} = -5, \ w_{22}=6 \ ]$.

We first convert the objective function to an Ising Hamiltonian using the following mapping:

$$x_i \to \frac{I_i - Z_i}{2}.$$

We call the objective function mapped to a Ising model a *QUBO Hamiltonian*:

$$ H_Q = \left[ - 4 (I - Z_1) (I - Z_2) - 5 (I - Z_1)^2 + 6 (I - Z_2)^2 \right]/4 \\[4mm] $$

which simplifies to

$$
H_Q = [ - 4 (I - Z_1 - Z_2 + Z_1Z_2) - 5 (I - 2Z_1 + Z_1^2) + 6 (I - 2Z_2 + Z_2^2) ] / 4 $$
$$ = [ - 4 (I - Z_1 - Z_2 + Z_1Z_2) - 5 (2I - 2Z_1) + 6 (2I - 2Z_2) ] / 4 $$
$$ = - \frac{1}{2} I + \frac{7}{2} Z_1 - 2 Z_2 - Z_1Z_2 $$

The problem can be solved by finding the minimum of

$$ \langle H_Q \rangle = \bra{\psi} \frac{7}{2} Z_1 - 2 Z_2 - Z_1Z_2 \ket{\psi} $$

which corresponds to the computational state $\ket{10}$ and equals to -5, including the offset term $-\frac{1}{2}I$.

The QUBO Hamiltonian is composed of single and two-qubit $Z$-Pauli terms. The weights of these terms are a linear combination of the weights $w$ of the QUBO objective function Q and can be easily derived.

We label them $\nu$. In this example, they are: $[ \ \nu_{22} = -2, \ \nu_{11} = 7/2, \ \nu_{12} = -1 \ ].$

## Formula

Any QUBO problem can be written, without loss of generality, in the following symmetric matricial form:

$$Q = \sum_{i=1}^N \sum_{j=1}^N w_{i,j} x_i x_j$$

This is possible because $x_i=x_i^2$ and because $x_i$ and $x_j$ commute.
Using the binary-to-Ising mapping defined above, we obtain the following Hamiltonian:

$$
H_Q = \sum_{i=1}^N \sum_{j=1}^N w_{i,j} \left( \frac{I_i - Z_i}{2} \right) \left( \frac{I_j - Z_j}{2} \right)\\[4mm]
=
\ \sum_{i=1}^N \sum_{j \ge i}^N \frac{w_{i,j}}{2} I \ \
- \ \ \sum_{i=1}^N Z_i \sum_{j=1}^N \frac{w_{i,j}}{2} \ \
+ \ \ \sum_{i=1}^N \sum_{j>i}^N \frac{w_{i,j}}{2} Z_i Z_j
$$

Therefore, we have:

$$
c = \sum_{i=1}^N \sum_{j \ge i}^N \frac{w_{i,j}}{2},
\qquad
Z_i : \nu_{i,i} = - \sum_{j=1}^N \frac{w_{i,j}}{2},
\qquad
Z_i Z_j : \nu_{i,j} = \frac{w_{i,j}}{2}
$$

Validation on the example above:

$$
H_Q \ \ = \ \ \frac{-5 -2 +6}{2} I \ \ - \ \ \left( Z_1 \frac{-5-2}{2} \ + \ Z_2 \frac{-2+6}{2} \right) \ \ + \ \ \frac{-2}{2} Z_1 Z_2\\[4mm]
= \ - \ \frac{1}{2} I \ + \ \frac{7}{2} Z_1 \ - \ 2 Z_2 \ - \ Z_1 Z_2\\[4mm]
$$
