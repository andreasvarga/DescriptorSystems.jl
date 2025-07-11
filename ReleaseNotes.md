# Release Notes

## Version 1.5.0

Minor release which provides extended functionality covering sparse matrix models and models with structured matrices. Most of the basic operations are supported for the new descriptor system types. Several functions such as `dss`, `gbalmr`, `gh2norm`, `ghanorm`, `glinfnorm`, `evalfr`, `freqresp`, `dcgain`, `gnrank`, have been extended to handle large scale sparse matrix models. The `SparseArrays` package has been included among the standard dependencies.

TODO: Implement basic connections for sparse matrix models. 

## Version 1.4.4

Some fixes to handle structured and sparse matrices in the `dss` function.   

## Version 1.4.3

Some fixes of errors originating from the enhanced definition of the `DescriptorStateSpace` object.   

## Version 1.4.2

Enhanced definition of the `DescriptorStateSpace` object.   

## Version 1.4.1

Patch release with enhanced functions for the computation of system norms. 

## Version 1.4

Version bump to comply with Julia 1.8 and higher. 

## Version 1.3.9

Patch release with new functions `gprescale`, `gbalqual` and `pbalqual` related to balancing of descriptor system models and several enhanced functions, such as: `gminreal`, `gdec`, `gzero`, `gzeroinfo`, to automatically perform balancing in the case of poorly scaled system models.  

## Version 1.3.8

Patch release with updated rss and rdss functions. 

## Version 1.3.7

Patch release to correct name conflict for the function order with Polynomials. 

## Version 1.3.6

Patch release to correct simplify and show for ratios of constants. 

## Version 1.3.5

Patch release enhancing step response computation. 

## Version 1.3.4

Patch release including a new function for step response computation. 

## Version 1.3.3

Patch release to enforce type stability in concatenation operations.

## Version 1.3.2

Patch release to enhance promotion for E matrices with I.

## Version 1.3.1

Patch release to eliminate warnings due to type piracy.

## Version 1.3 

Minor release to alleviate/fix hcat/vcat/hvcat type piracy issues in Julia 1.6 and 1.7 and to use Polynomials 3.0. Internal type piracy with MatrixPencils has been completely eliminated. 

## Version 1.2.2 

Patch release to correct a bug in `glasol`.

## Version 1.2.1 

Patch release to overcome a bug in `orghr!` for null dimension [#43680](https://github.com/JuliaLang/julia/issues/43680) and correct minor bugs in `rss` and `rdss` to handle null order systems. The functions `glasol` and `grasol` have been enhanced to reliably detect Nehari problems.

## Version 1.2.0

This minor release provides some new functions and enhancements as follows:

* three new functions for manipulating subsystems of descriptor systems: `dssubset`, to set a subsystem equal to a given system, `dszeros` to set a subsystem equal to zero, and `dssubsel` to select a subsystem corresponding to a specified zero-nonzero pattern;
* a new function `dsdiag` has been implemented to build a `k`-times diagonal concatenation of a descriptor system;
* a new function `feedback` has been implemented to apply a static output feedback gain to a descriptor system;
* the `*`, `+` and `-` operations of two descriptor systems covers now the left/right 
multiplication/addition/substraction with scalar systems;
* the functions `glinfldp` and `gnehari` have been enhanced to handle cases with poles on the boundary of the stability domain;
* the function `gsvselect` has been renamed as `dsxvarsel`;
* some bugs fixed.

## Versions 1.1.2 and 1.1.3

Patch releases to fix a bug and to address issue [#8](https://github.com/andreasvarga/DescriptorSystems.jl/issues/8). 

## Version 1.1.1

This patch release uses a new preprocessing routine for reduction of system matrices to Hessenberg forms in the computation of frequency responses.  

## Version 1.1.0

This minor release relies on release `v2.0` (or higher) of `MatrixEquations.jl`, which employs `LinearMaps.jl` (instead of `LinearOperators.jl` in previous releases). 

## Version 1.0.1

This patch release only improves the demonstration script `DSToolsDemo.jl`.

## Version 1.0.0

This is the first major release which implements the targeted functionality available in the companion [`DSTOOLS`](https://github.com/andreasvarga/DescriptorSystemTools) toolbox developed for MATLAB. The latest additions include new functions for conversion of descriptor systems to standard form with determination of consistent initial state and state mapping matrices, discretization of descriptor systems and rational transfer functions, conversion to input-output form (rational or polynomial transfer function matrix), computation of generalized inverses and determination of time responses. Also a function has been implemented
to compute the left and right projection matrices associated with the computation of minimal realizations.  Several bug fixes have been also performed. 

## Version 0.7.0

This minor release relies on a new definition of the _RationalTransferFunction_ type as a subtype of the _RationalFunction_ type available in the Polynomials package starting with the version v2.0.9. Also, the _DescriptorStateSpace_ type has been simplified. Several bug fixes have been also performed. 

## Version 0.6.0

This minor release includes new functions for solving model matching problems, such as the Nehari approximation problems, least-distance problems, and approximate solution  of linear equations with rational matrices. The definition of the descriptor system object has been simplified to have only fields with concrete types. Several bug fixes have been also performed. 

## Version 0.5.0

This minor release includes new functions for advanced manipulation of transfer function matrices via their descriptor system realizations, such as, the computation of range and coimage, the computation of normalized coprime factorizations, the computation of special spectral factorizations, the efficient computation of the frequency responses and the computation of the ν-gap distance between two systems. The definition of the rational transfer function object has been simplified to have only fields with concrete types.  

## Version 0.4.3

This patch version alleviates the excessive compilation times arising after updating the package to using the latest version v2.0 of Polynomials.jl.

## Version 0.4.2

Patch release to enhance the definition of the rational transfer function object to better fit to 
Polynomials.jl v2.0.

## Version 0.4.1

Patch release to upgrade to using Polynomials.jl v2.0 and MatrixPencils 1.6.2.

## Version 0.4.0

This minor release includes new functions for advanced manipulation of transfer function matrices via their descriptor system realizations,
such as, the computation of additive spectral decompositions, the computation of nullspace bases, the solution of linear equations with rational matrices and the solution of minimal cover problems.

## Version 0.3.0

This minor release defines a new system theoretical object, the _rational transfer function_, and includes new functions which support the operations involving rational functions. Full support is also provided to build matrices with rational transfer function elements and constructors are provided to compute descriptor system realizations of rational matrices. Additionally, a function to convert a descriptor system description to a standard one and a second function to perform general bilinear transformation of descriptor systems have been implemented. A function for the generation of the transfer functions of some commonly used bilinear transformations (Cayley, Tustin, Moebius, etc.) is also provided.

## Version 0.2.0

This minor release includes new functions for the analysis of descriptor systems, such as for computation of poles and zeros,
evaluation of normal rank, evaluation of several system norms, as well as a collection of new functions for some basic coprime and inner-outer type factorizations.

## Version 0.1.0

This is the initial release providing a prototype implementation of the basic descriptor system object, jointly with functions to perform basic operations, connections and simplifications of descriptor system models.
