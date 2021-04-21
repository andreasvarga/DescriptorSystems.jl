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

It is well known that the descriptor system representation is the most general description for a linear time-invariant system. Continuous-time descriptor systems arise frequently from modelling interconnected systems containing algebraic loops or constrained mechanical systems which describe contact phenomena. Discrete-time descriptor representations are frequently used to model economic processes. A main apeal of descriptor system models is that the manipulation of rational and polynomial matrices can be easily performed via their descriptor system representations, since each rational or polynomial matrix can be interpreted as the TFM of a descriptor system. For an introductory presentation of the main concepts, see [1].

The theoretical background for the analysis of descriptor systems closely relies on investigating the properties of certain linear matrix pencils, as the regular _pole pencil_ `P(λ) = A-λE`, or the generally singular _system matrix pencil_ `S(λ) = [A-λE B; C D]`. Therefore, the main analysis tools of descriptor systems are pencil manipulation techniques (e.g., reductions to various Kronecker-like forms), as available in the [MatrixPencils](https://github.com/andreasvarga/MatrixPencils.jl) package [2]. Among the main applications of pencil manipulation algorithms, we mention  the computation of minimal nullspace bases, the computation of poles and zeros, the determination of the normal rank of polynomial and rational matrices, computation of various factorizations of rational matrices, as well as the solution of linear equations with polynomial or rational matrices. Important additional computational ingredients in these applications are tools for solving matrix equations, as various Lyapunov, Sylvester and Riccati equations. These tools are provided by the [MatrixEquations](https://github.com/andreasvarga/MatrixEquations.jl) package [3].

The available functions in the `DescriptorSystems.jl` package cover both standard and descriptor systems with real or complex coefficient matrices. The current version of the package includes the following functions:

**Building descriptor system state-space models**

* **[`dss`](@ref)**  Construction of descriptor state-space models.
* **[`dssdata`](@ref)**   Extraction of matrix-data from a descriptor state-space model.

**Building rational transfer functions**

* **[`RationalTransferFunction`](@ref)**  Construction of rational transfer function objects.
* **[`rtf`](@ref)**  Building rational transfer functions.

**Interconnecting descriptor system models**

* **[`append`](@ref)**  Building aggregate models by appending the inputs and outputs.
* **[`parallel`](@ref)**   Connecting models in parallel (also overloaded with **`+`**).
* **[`series`](@ref)**   Connecting models in series (also overloaded with **`*`**).
* **[`horzcat`](@ref)**   Horizontal concatenation of descriptor system models (also overloaded with **`[ * * ]`**).
* **[`vertcat`](@ref)**   Vertical concatenation of descriptor system models (also overloaded with **`[ *; * ]`**).

**Basic operations on descriptor system models**

* **[`inv`](@ref)**  Inversion of a system.
* **[`ldiv`](@ref)**   Left division for two systems (also overloaded with **`\`**).
* **[`rdiv`](@ref)**   Right division for two systems (also overloaded with **`/`**).
* **[`gdual`](@ref)**   Building the dual of a descriptor system (also overloaded with **`transpose`**)
* **[`ctranspose`](@ref)**  Building the conjugate transpose of a system (also overloaded with **`adjoint`** and **`'`**).
* **[`adjoint`](@ref)**  Building the adjoint of a system.
* **[`gbilin`](@ref)**  Generalized bilinear transformation of a descriptor system.

**Some operations on rational transfer functions and matrices**

* **[`simplify`](@ref)**  Pole-zero cancellation.
* **[`normalize`](@ref)**   Normalization of a rational transfer function to monic denominator.
* **[`confmap`](@ref)**   Applying a conformal mapping transformation to a rational transfer function.
* **[`rmconfmap`](@ref)**   Applying a conformal mapping transformation to a rational transfer function matrix.
* **[`zpk`](@ref)**  Computation of zeros, poles and gain of a rational transfer function.
* **[`rtfbilin`](@ref)**  Generation of common bilinear transformations and their inverses.

**Simplification of descriptor system models**

* **[`gminreal`](@ref)**  Minimal realization of descriptor systems.
* **[`gir`](@ref)**   Irreducible realization of descriptor systems.
* **[`gbalmr`](@ref)**   Reduced-order approximations of descriptor systems using balancing related methods.
* **[`gss2ss`](@ref)**   Conversion to SVD-like forms without non-dynamic modes.

**Descriptor system analysis**

* **[`isregular`](@ref)** Test whether a descriptor system has a regular pole pencil.
* **[`gpole`](@ref)**    Poles of a descriptor system.
* **[`gpoleinfo`](@ref)**   Poles and pole structure information of a descriptor system.
* **[`isproper`](@ref)**   Test whether a descriptor system is proper.
* **[`isstable`](@ref)**   Test whether a descriptor system is stable.
* **[`gzero`](@ref)**  Zeros of a descriptor system.
* **[`gzeroinfo`](@ref)** Zeros and zero structure information of a descriptor system.
* **[`gnrank`](@ref)**  Normal rank of the transfer function matrix of a descriptor system.
* **[`ghanorm`](@ref)**  Hankel norm of a proper and stable descriptor system.
* **[`gl2norm`](@ref)**  `L2` norm of a descriptor system.
* **[`gh2norm`](@ref)**  `H2` norm of a descriptor system.
* **[`glinfnorm`](@ref)**  `L∞` norm of a descriptor system.
* **[`ghinfnorm`](@ref)**  `H∞` norm of a descriptor system.
* **[`gnugap`](@ref)**  `ν-gap` distance between two descriptor systems.

**Factorization of descriptor systems**

* **[`grcf`](@ref)**  Right coprime factorization with proper and stable factors.
* **[`glcf`](@ref)**   Left coprime factorization with proper and stable factors.
* **[`grcfid`](@ref)**   Right coprime factorization with inner denominator.
* **[`glcfid`](@ref)**   Left coprime factorization with inner denominator.
* **[`gnrcf`](@ref)**  Normalized right coprime factorization.
* **[`gnlcf`](@ref)**   Normalized left coprime factorization.
* **[`giofac`](@ref)**   Inner-outer/QR-like factorization.
* **[`goifac`](@ref)**   Co-outer-co-inner/RQ-like factorization.
* **[`grsfg`](@ref)**   Right spectral factorization of `γ^2*I-G'*G`. 
* **[`glsfg`](@ref)**   Left spectral factorization of `γ^2*I-G*G'`. 

**Advanced operations on transfer function matrices**

* **[`gsdec`](@ref)**  Additive spectral decompositions.
* **[`grnull`](@ref)**   Right nullspace basis of a transfer function matrix.
* **[`glnull`](@ref)**   Left nullspace basis of a transfer function matrix.
* **[`grange`](@ref)**   Range space basis of a transfer function matrix. 
* **[`gcrange`](@ref)**  Coimage space basis of a transfer function matrix. 
* **[`grsol`](@ref)**   Solution of the linear rational matrix equation `G(λ)*X(λ) = F(λ)`.
* **[`glsol`](@ref)**   Solution of the linear rational matrix equation `X(λ)*G(λ) = F(λ)`.
* **[`grmcover1`](@ref)**  Right minimum dynamic cover of Type 1 based order reduction.
* **[`glmcover1`](@ref)**   Left minimum dynamic cover of Type 1 based order reduction.
* **[`grmcover2`](@ref)**  Right minimum dynamic cover of Type 2 based order reduction.
* **[`glmcover2`](@ref)**  Left minimum dynamic cover of Type 2 based order reduction.

**Solution of model-matching problems**

* **[`gnehari`](@ref)**  Generalized Nehari approximation.
* **[`glinfldp`](@ref)**  Solution of the least distance problem.
* **[`grasol`](@ref)**   Approximate solution of the linear rational matrix equation `G(λ)*X(λ) = F(λ)`.
* **[`glasol`](@ref)**   Approximate solution of the linear rational matrix equation `X(λ)*G(λ) = F(λ)`.

## Future plans

The targeted v1.0 will additionally include functions for several basic conversions to/from input-output representations as well as functions for time-response and frequency response computation.
Later future developments will address support for several new classes of generalized LTI systems types and for polynomial system models.

## [Release Notes](https://github.com/andreasvarga/DescriptorSystems.jl/blob/main/ReleaseNotes.md)

## Main developer

[Andreas Varga](https://sites.google.com/view/andreasvarga/home)

License: MIT (expat)

## References

[1]   A. Varga, Solving Fault Diagnosis Problems – Linear Synthesis Techniques, Vol. 84 of
Studies in Systems, Decision and Control, Springer International Publishing, 2017.

[2]  A. Varga, [MatrixPencils.jl: Matrix pencil manipulation using Julia](https://github.com/andreasvarga/MatrixPencils.jl).
[Zenodo: https://doi.org/10.5281/zenodo.3894503](https://doi.org/10.5281/zenodo.3894503).

[3]  A. Varga, [MatrixEquations.jl: Solution of Lyapunov, Sylvester and Riccati matrix equations using Julia](https://github.com/andreasvarga/MatrixEquations.jl). [Zenodo: https://doi.org/10.5281/zenodo.3556867](https://doi.org/10.5281/zenodo.3556867).
