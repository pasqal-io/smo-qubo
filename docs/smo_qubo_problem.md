# Space Mission Optimization

## Problem Description

The problem is to acquire as many images as possible while obeying the attitude maneuvering constraint of the satellite.
Specifically, we examine the agile Earth observation satellite (AEOS) scheduling problem (AEOSSP), where the time to solution grows exponentially with the problem size.

Heuristic methods are employed to address the AEOSSP in industrial settings since solutions need to be obtained in a short finite time frame.

### Cost function
---

The cost function of the problem is:

$$ C = - \sum_{r \in R} \sum_{i \in I_r} x_{r,i} $$

with:

| <div style="width:90px">Variable</div> | <div style="width:80px">Meaning</div> |
| - | - |
| $R$ | set of acquisition requests |
|$r \in R$ | acquisition request |
| $I_r$ | list of imaging attempts for a request $r$, resulting from the discretization of the access period, ordered by increasing time |
| $i ∈ I_r$ | imaging attempt |
| $x_{r,i} \in \{0,1\}$ | binary variable: selection $(x=1)$ or dismissal $(x=0)$ |

<br/>

### Constraints
---

1. No more than one selected imaging attempt per acquisition request

$$ \sum_{i\in I_r} x_{r,i} \le 1 \qquad \forall r \in R $$

2. Mutual exclusion of certain pairs of imaging attemps

$$ x_{r,i} + x_{s,j} \le 1 \qquad \forall (i,j) \in F_{r,s} \quad \land \quad \forall (r,s) \in R^2, \quad r \ne s $$

where

$$ F_{r,s} = \{ (i,j) \in I_r \times I_s : t_i^r \le t_j^s < t_i^r + d_{r,i}^{\mathrm{acq}} + d_{(r,i) \to (s,j)}^{\mathrm{man}} \} $$

## QUBO Formulation

The total QUBO objective function is:

$$Q = C + \lambda_u C_u + \lambda_t C_t$$

where $C$ is the problem cost function defined above, and constraints 1. and 2. are fulfilled when the two penalty terms vanish:

$$C_u = \sum_{r \in R} \sum_{(i,j) \in I_r}^{i \ne j} x_{r,i} x_{s,j} \qquad | \qquad C_t = \sum_{r,s \in R}^{r \ne s} \sum_{(i,j) \in F_{r,s}} x_{r,i} x_{s,j}$$

In order to make sure that the penalty terms vanish in the optimal solution of the problem, the penalty weights $(\lambda_u, \ \lambda_t)$ ought to be chosen large enough.

But how small can they be, before losing effect? An upper bound for the minimum sufficient penalty weights is given by the value of the aquistions in the problem (here equal to 1), meaning we can choose:

$$ \lambda_u, \ \lambda_t \ \gtrsim \ 1 $$

# Legend

#### Equations

| <div style="width:110px">Name</div> | <div style="width:80px">Equation</div> |
| - | - |
| Cost function | $$Q = C + \lambda_u C_u + \lambda_t C_t$$ |
| Constraint 1. | $$C_u = \sum_{r \in R} \sum_{(i,j) \in I_r}^{i \ne j} x_{r,i} x_{s,j}$$ |
| Constraint 2. | $$C_t = \sum_{r,s \in R}^{r \ne s} \sum_{(i,j) \in F_{r,s}} x_{r,i} x_{s,j}$$ |


#### Legend

| <div style="width:70px">Symbol</div> | <div style="width:80px">Description</div> |
| - | - |
| $\lambda_u$ | penalty weight, single acquisition request |
| $\lambda_t$ | penalty weight, acquisition pairs mutual exclusion |

---

#### Domain

| <div style="width:110px">Variable</div> | <div style="width:80px">Meaning</div> |
| - | - |
| $x_{r,i} \in \{0,1\}$ | $\forall i \in I_r \quad \land \quad \forall r \in R$ |
