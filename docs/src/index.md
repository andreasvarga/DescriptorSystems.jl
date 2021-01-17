```@meta
CurrentModule = DescriptorSystems
DocTestSetup = quote
    using DescriptorSystems
end
```

# DescriptorSystems.jl

[![DocBuild](https://github.com/andreasvarga/MatrixPencils.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/MatrixPencils.jl/actions)
[![Code on Github.](https://img.shields.io/badge/code%20on-github-blue.svg)](https://github.com/andreasvarga/DescriptorSystems.jl)

A descriptor system is a generalized state-space representation of the form

    Eλx(t) = Ax(t) + Bu(t),
    y(t)   = Cx(t) + Du(t),

where `x(t)` is the state vector, `u(t)` is the input vector, and `y(t)` is the
output vector, and where `λ` is either the differential operator `λx(t) = dx(t)/dt`  for a continuous-time system or the advance operator `λx(t) = x(t + ΔT)` for a discrete-time system with the sampling time `ΔT`.
In all what follows, we assume `E` is square and possibly singular, and the pencil `A − λE` is regular (i.e.,
`det(A − λE) ̸≡ 0`). If `E = I`, we call the above representation a  _standard_ state-space system.

The corresponding input-output representation is

    Y(λ) = G(λ)U(λ),

where, depending on the system type, `λ = s`, the complex variable in the Laplace transform for
a continuous-time system, or `λ = z`, the complex variable in the `Z`-transform for a discrete-time
system, `Y(λ)` and `U(λ)` are the Laplace- or `Z`-transformed output and input vectors, respectively,
and `G(λ)` is the rational _transfer function matrix_ (TFM) of the system, defined as

                    -1
    G(λ) = C(λE − A)  B + D.

It is well known that the descriptor system representation is the most general description for a linear time-invariant system. Continuous-time descriptor systems arise frequently from modelling interconnected systems containing algebraic loops or constrained mechanical systems which describe contact phenomena. Discrete-time descriptor representations are frequently used to model economic processes. A main apeal of descriptor system models is that the manipulation of rational and polynomial matrices can be easily performed via their descriptor system representations, since each rational or polynomial matrix can be interpreted as the TFM of a descriptor
system. For an introductory presentation of the main concepts, see [1].

The theoretical background for the analysis of descriptor systems closely relies on investigating the properties of certain linear matrix pencils, as the regular _pole pencil_ `P(λ) = A-λE`, or the generally singular _system matrix pencil_ `S(λ) = [A-λE B; C D]`. Therefore, the main analysis tools of descriptor systems are pencil manipulation techniques (e.g., reductions to various Kronecker-like forms), as available in the [MatrixPencils](https://github.com/andreasvarga/MatrixPencils.jl) package [2]. Among the main applications of pencil manipulation algorithms, we mention  the computation of minimal nullspace bases, the computation of poles and zeros, the determination of the normal rank of polynomial and rational matrices, computation of various factorizations of rational matrices, as well as the solution of linear equations with polynomial or rational matrices. Important additional computational ingredients in these applications are tools for solving matrix equations, as various Lyapunov, Sylvester and Riccati equations. These tools are provided by the [MatrixEquations](https://github.com/andreasvarga/MatrixEquations.jl) package [3].

The available functions in the `DescriptorSystems.jl` package cover both standard and descriptor systems with real or complex coefficient matrices. The current version of the package includes the following functions:

**Building descriptor system state-space models**

* **dss**  Construction of descriptor state-space models.
* **dssdata**   Extraction of matrix-data from a descriptor state-space model.

**Interconnecting descriptor system models**

* **append**  Building aggregate models by appending the inputs and outputs.
* **parallel**   Connecting models in parallel (also overloaded with **`+`**).
* **series**   Connecting models in series (also overloaded with **`*`**).
* **horzcat**   Horizontal concatenation of descriptor system models (also overloaded with **`[ * * ]`**).
* **vertcat**   Vertical concatenation of descriptor system models (also overloaded with **`[ *; * ]`**).

**Basic operations on descriptor system models**

* **inv**  Inversion of a descriptor system.
* **ldiv**   Left division for two descriptor systems (also overloaded with **`\`**).
* **rdiv**   Right division for two descriptor systems (also overloaded with **`/`**).
* **gdual**   Construction of the dual of a descriptor system (also overloaded with **`transpose`**)
* **ctranspose**  Construction of the conjugate transpose of a descriptor system (also overloaded with **`'`**).

**Simplification of descriptor system models**

* **gminreal**  Minimal realization of descriptor systems.
* **gir**   Irreducible realization of descriptor systems.
* **gbalmr**   Reduced-order approximations of descriptor systems using balancing related methods.

## Future plans

This is a rapidly evolving software project for which new functionality will be frequently added.

## [Release Notes](https://github.com/andreasvarga/DescriptorSystems.jl/blob/master/ReleaseNotes.md)

## Main developer

[Andreas Varga](https://sites.google.com/view/andreasvarga/home)

License: MIT (expat)

## References

[1]   A. Varga, Solving Fault Diagnosis Problems – Linear Synthesis Techniques, Vol. 84 of
Studies in Systems, Decision and Control, Springer International Publishing, 2017.

[2]  A. Varga, [MatrixPencils.jl: Matrix pencil manipulation using Julia](https://github.com/andreasvarga/MatrixPencils.jl).
[Zenodo: https://doi.org/10.5281/zenodo.3894503](https://doi.org/10.5281/zenodo.3894503).

[3]  A. Varga, [MatrixEquations.jl: Solution of Lyapunov, Sylvester and Riccati matrix equations using Julia](https://github.com/andreasvarga/MatrixEquations.jl). [Zenodo: https://doi.org/10.5281/zenodo.3556867](https://doi.org/10.5281/zenodo.3556867).
