# DescriptorSystems.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4568159.svg)](https://doi.org/10.5281/zenodo.4568159)
[![DocBuild](https://github.com/andreasvarga/DescriptorSystems.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/DescriptorSystems.jl/actions)
[![codecov.io](https://codecov.io/gh/andreasvarga/DescriptorSystems.jl/coverage.svg?branch=main)](https://codecov.io/gh/andreasvarga/DescriptorSystems.jl?branch=main)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://andreasvarga.github.io/DescriptorSystems.jl/dev/)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/DescriptorSystems.jl/blob/main/LICENSE.md)

## Manipulation of generalized state-space (descriptor) system representations

## Compatibility

Julia 1.5 and higher.

## How to install

````JULIA
pkg> add DescriptorSystems
pkg> test DescriptorSystems
````

## About

`DescriptorSystems.jl` is a collection of Julia functions for numerical computations related to generalized state-space (descriptor) system representations in the continuous-time form

     Edx(t)/dt = Ax(t) + Bu(t) ,
     y(t)      = Cx(t) + Du(t) ,

or in the discrete-time form

     Ex(t+1) = Ax(t) + Bu(t) ,
     y(t)    = Cx(t) + Du(t) ,

where `x(t)`, `u(t)` and `y(t)` are the system state vector, system input vector and system output vector, respectively, and `t` is the continuous or discrete time variable.  

This collection also allows the operation on and manipulation of rational and polynomial matrices via their
descriptor system realizations.

The functionality of many of the implemented functions parallel or even extend the functionality of the
Control System Toolbox of MATLAB and is similar to that of the [`DSTOOLS`](https://github.com/andreasvarga/DescriptorSystemTools) collection of tools. The underlying computational functions are based on the
[`MatrixPencils.jl`](https://github.com/andreasvarga/MatrixPencils.jl) and
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl) packages.

Many of the functions implement the computational procedures described in Chapter 10 of the book:

* Andreas Varga, "[Solving Fault Diagnosis Problems - Linear Synthesis Techniques](http://www.springer.com/us/book/9783319515588)", vol. 84 of Studies in Systems, Decision and Control, Springer International Publishing, xxviii+394, 2017.

This book provides additional information on the mathematical background on rational matrices and descriptor systems, and gives detailed descriptions of most of the underlying procedures.

The current version of the package includes the following categories of functions:

* Building descriptor system state-space models

* Building rational transfer function input-output models

* Basic operations on descriptor system models

* Basic conversions of descriptor system models, including discretization

* Interconnecting descriptor system models

* Simplification of descriptor system models

* Descriptor system analysis

* Factorization of descriptor system transfer function matrices

* Advanced operations on transfer function matrices via their descriptor system realizations

* Solution of model-matching problems

* Time response computation

## Future plans

Future developments will address support for several new classes of generalized LTI systems types (e.g., for polynomial system models).

## Supplementary information

Supplementary information on the existing functionality is also available on [arXiv](https://arxiv.org/abs/1707.07140) in the documentation of the companion Matlab [`DSTOOLS`](https://github.com/andreasvarga/DescriptorSystemTools) collection.
