# IPE Algorithms and Mathematical Formulations

## Overview

The Interactionist Phylogeny Engine (IPE) implements a comprehensive suite of evolutionary and physiological algorithms. This document provides detailed mathematical formulations and algorithmic descriptions for the core computational methods.

## Table of Contents

- [State Space Dynamics](#state-space-dynamics)
- [Evolutionary Algorithms](#evolutionary-algorithms)  
- [Physiological Game Theory](#physiological-game-theory)
- [Environmental Interaction Models](#environmental-interaction-models)
- [Selection and Fitness Functions](#selection-and-fitness-functions)
- [Population Dynamics](#population-dynamics)
- [Mutation and Variation](#mutation-and-variation)

## State Space Dynamics

### Physiological State Vector

Each organism is represented by a state vector **x** ∈ ℝ^n in physiological trait space:

```
x = [x₁, x₂, ..., xₙ]ᵀ
```

Where each component xᵢ represents a physiological trait:
- x₁: Relative heart mass (dimensionless)
- x₂: Lung diffusion capacity (ml/min/mmHg/kg)
- x₃: Hematocrit (fraction)
- x₄: Muscle fiber oxidative capacity (μmol/min/g)
- ... (additional traits as needed)

### State Space Constraints

Physiological constraints impose bounds on the state space:

```
C(x) = {x ∈ ℝⁿ : gᵢ(x) ≤ 0, i = 1,...,m}
```

**Allometric constraints:**
```
g₁(x) = x₁ - α₁M^(β₁-1) ≤ 0    (heart mass scaling)
g₂(x) = x₂ - α₂M^(β₂-1) ≤ 0    (lung capacity scaling)
```

Where M is body mass, and αᵢ, βᵢ are allometric constants.

**Thermodynamic constraints:**
```
g₃(x) = Σᵢ cᵢxᵢ - E_max ≤ 0    (energy budget constraint)
```

Where cᵢ is the energetic cost of trait xᵢ and E_max is the maximum available energy.

### State Transitions

State evolution follows the differential equation:

```
dx/dt = f(x, E, P, t) + σ(x)ξ(t)
```

Where:
- f(x, E, P, t): Deterministic dynamics (selection, growth)
- E: Environmental parameters
- P: Population state
- σ(x)ξ(t): Stochastic perturbations (mutation, noise)
- ξ(t): White noise process

## Evolutionary Algorithms

### Quantitative Genetics Model

Trait evolution follows the multivariate breeder's equation:

```
Δz̄ = GP⁻¹S
```

Where:
- Δz̄: Change in mean trait values
- G: Genetic covariance matrix
- P: Phenotypic covariance matrix  
- S: Selection gradient vector

**Genetic covariance matrix:**
```
G = h²P
```

Where h² is the heritability matrix.

**Selection gradient:**
```
S = ∇ₓW̄(x̄)
```

Where W̄(x̄) is mean population fitness.

### Individual-Based Evolution

For individual-based simulations, offspring traits are generated as:

```
x'ᵢ = xᵢ + μᵢ + εᵢ
```

Where:
- xᵢ: Parent trait vector
- μᵢ: Mutation vector ~ N(0, Σₘ)
- εᵢ: Environmental noise ~ N(0, Σₑ)

**Mutation covariance matrix:**
```
Σₘ = uMₘ
```

Where u is the mutation rate and Mₘ is the mutation effect matrix.

## Physiological Game Theory

### Game Formulation

Physiological traits are treated as strategies in evolutionary games. For n players (physiological systems), the payoff for system i is:

```
πᵢ(sᵢ, s₋ᵢ, E) = Bᵢ(sᵢ, E) - Cᵢ(sᵢ, s₋ᵢ) - Tᵢ(sᵢ, s₋ᵢ)
```

Where:
- sᵢ: Strategy of system i (trait values)
- s₋ᵢ: Strategies of other systems
- Bᵢ(sᵢ, E): Environmental benefits
- Cᵢ(sᵢ, s₋ᵢ): Direct costs
- Tᵢ(sᵢ, s₋ᵢ): Trade-off costs

### Hypoxia Allocation Game

For oxygen allocation between heart, lungs, and blood:

**Strategy space:** S = {s = (s₁, s₂, s₃) : s₁ + s₂ + s₃ = 1, sᵢ ≥ 0}

**Payoffs:**
```
π₁(s, E) = α₁s₁√(O₂_delivery) - β₁s₁²        (heart)
π₂(s, E) = α₂s₂√(O₂_uptake) - β₂s₂²          (lungs)  
π₃(s, E) = α₃s₃√(O₂_transport) - β₃s₃²       (blood)
```

**Oxygen delivery function:**
```
O₂_delivery = min(s₂·V̇_O₂max, s₃·[Hb]·1.34, s₁·Q̇max)
```

Where:
- V̇_O₂max: Maximum oxygen uptake
- [Hb]: Hemoglobin concentration
- Q̇max: Maximum cardiac output

### Nash Equilibrium Solution

The evolutionary stable strategy satisfies:

```
∂πᵢ/∂sᵢ|_{s*} = λ    for all i where s*ᵢ > 0
∂πᵢ/∂sᵢ|_{s*} ≤ λ    for all i where s*ᵢ = 0
```

Where λ is the Lagrange multiplier for the constraint Σsᵢ = 1.

## Environmental Interaction Models

### Altitude-Oxygen Relationship

Atmospheric oxygen availability follows:

```
pO₂(h) = pO₂₀ · e^(-h/h₀)
```

Where:
- pO₂₀: Sea-level oxygen partial pressure (159 mmHg)
- h: Altitude (m)  
- h₀: Scale height (≈ 8400 m)

### Temperature Effects on Metabolism

Metabolic rate temperature dependence:

```
MR(T) = MR₀ · e^(E_a/k(1/T₀ - 1/T))
```

Where:
- MR₀: Reference metabolic rate
- E_a: Activation energy
- k: Boltzmann constant
- T₀: Reference temperature (K)
- T: Current temperature (K)

### Hypoxic Stress Function

Oxygen stress experienced by an organism:

```
S_O₂(x, E) = max(0, O₂_demand(x) - O₂_supply(x, E))
```

**Oxygen demand:**
```
O₂_demand(x) = BMR(x) + AMR(x) + GMR(x)
```

Where BMR, AMR, GMR are basal, active, and growth metabolic rates.

**Oxygen supply:**
```
O₂_supply(x, E) = min(V̇_O₂max(x₂), Q̇max(x₁)·CaO₂(x₃))
```

Where CaO₂ is arterial oxygen content.

## Selection and Fitness Functions

### Fitness Landscape

Individual fitness is defined as:

```
W(x, E, P) = W₀ · e^(-β·S(x,E)) · R(x, P)
```

Where:
- W₀: Base fitness
- β: Selection strength parameter
- S(x, E): Environmental stress function
- R(x, P): Frequency-dependent selection

### Environmental Stress Function

```
S(x, E) = Σᵢ wᵢ · max(0, Dᵢ(x, E) - Tᵢ)
```

Where:
- wᵢ: Weight of stressor i
- Dᵢ(x, E): Demand for function i
- Tᵢ: Tolerance threshold for stressor i

### Frequency-Dependent Selection

Competition effects:

```
R(x, P) = (1 + α·∫ K(x, x')n(x')dx')^(-1)
```

Where:
- α: Competition strength
- K(x, x'): Competition kernel
- n(x'): Population density at trait x'

**Gaussian competition kernel:**
```
K(x, x') = e^(-||x-x'||²/(2σ²_comp))
```

## Population Dynamics

### Continuous Population Model

Population density n(x, t) evolves according to:

```
∂n/∂t = ∇·(D∇n) + r(x, E, P)n - d(x, E, P)n - ∇·(v(x, E, P)n)
```

Where:
- D: Diffusion tensor (mutation)
- r(x, E, P): Birth rate
- d(x, E, P): Death rate
- v(x, E, P): Drift velocity (selection)

### Discrete Generation Model

For discrete generations:

```
n_{t+1}(x) = ∫ W(x', E, P_t) · M(x|x') · n_t(x') dx'
```

Where M(x|x') is the mutation kernel from parent x' to offspring x.

### Carrying Capacity

Logistic population regulation:

```
r_eff(x, P) = r_max(x) · (1 - N_total/K(E))
```

Where:
- r_max(x): Intrinsic growth rate
- N_total: Total population size
- K(E): Environmental carrying capacity

## Mutation and Variation

### Multivariate Mutation

Mutation effects follow multivariate normal distribution:

```
μ ~ N(0, u·M)
```

**Mutation matrix structure:**
```
M = D^(1/2) · R · D^(1/2)
```

Where:
- D: Diagonal matrix of trait-specific mutation variances
- R: Correlation matrix between traits

### Mutational Constraints

Developmental constraints limit mutation directions:

```
P(μ) ∝ e^(-μᵀG⁻¹μ/2)
```

Where G is the genetic constraint matrix.

### Evolvability

Mutational variance in the direction of selection:

```
e(G, β) = βᵀGβ/|β|²
```

Where β is the selection gradient vector.

## Numerical Implementation

### Finite Difference Methods

State space derivatives approximated using:

```
∂f/∂xᵢ ≈ (f(x + hᵢeᵢ) - f(x - hᵢeᵢ))/(2hᵢ)
```

### Stochastic Integration

Evolution equations integrated using Euler-Maruyama method:

```
X_{t+Δt} = X_t + f(X_t, t)Δt + σ(X_t)ΔW_t
```

Where ΔW_t ~ N(0, Δt) is the Wiener increment.

### Monte Carlo Methods

Population statistics estimated via sampling:

```
⟨f(X)⟩ ≈ (1/N) Σᵢ f(Xᵢ)
```

With confidence intervals:

```
CI = ⟨f(X)⟩ ± t_{α/2} · SE(f)
```

## Computational Complexity

### Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Fitness evaluation | O(N·K) | O(N) |
| Mutation generation | O(N·n²) | O(n²) |
| Selection | O(N log N) | O(N) |
| Population update | O(N·n) | O(N·n) |

Where:
- N: Population size
- n: Number of traits
- K: Number of environmental factors

### Optimization Techniques

**Vectorization:**
```python
# Vectorized fitness calculation
fitness = np.exp(-beta * stress_function(population, environment))
```

**Parallel Processing:**
```python
# Parallel mutation generation
with multiprocessing.Pool() as pool:
    mutations = pool.map(generate_mutation, parent_traits)
```

**GPU Acceleration:**
```python
# CuPy-based state space operations
import cupy as cp
state_space_gpu = cp.array(state_space)
fitness_gpu = cp.exp(-cp.sum(stress_terms, axis=1))
```

## Validation and Testing

### Unit Tests

Each algorithm component is tested against:
- Analytical solutions (where available)
- Published benchmarks
- Simplified test cases with known outcomes

### Integration Tests

Full simulation pipelines tested for:
- Conservation laws (mass, energy)
- Boundary condition handling
- Long-term stability
- Parameter sensitivity

### Scientific Validation

Algorithm outputs compared with:
- Empirical data from literature
- Laboratory experimental results
- Field study observations
- Other simulation platforms

This mathematical framework provides the foundation for IPE's evolutionary and physiological modeling capabilities, ensuring both biological realism and computational efficiency.