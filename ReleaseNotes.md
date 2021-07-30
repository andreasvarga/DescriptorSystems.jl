# Release Notes

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
