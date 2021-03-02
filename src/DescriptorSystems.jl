module DescriptorSystems
# Release V0.1

using LinearAlgebra
using MatrixEquations
using MatrixPencils
using Polynomials
using Random
using Compat

import LinearAlgebra: BlasFloat, BlasReal, BlasComplex, copy_oftype, transpose, adjoint, opnorm, normalize
import Base: +, -, *, /, \, (==), (!=), ^, isapprox, iszero, convert, promote_op, size, length, ndims, 
             hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show, one, zero, eltype
import MatrixPencils: isregular, rmeval
import Polynomials: variable, degree

export DescriptorStateSpace, AbstractDescriptorStateSpace, dss, dssdata, rdss, rss, iszero, order
export AbstractRationalTransferFunction, RationalTransferFunction, rtf
export gminreal, gir, gbalmr, gsvselect, gss2ss, gbilin
export confmap, rmconfmap, simplify, normalize, poles, gain, zpk, rtfbilin, numpoly, denpoly, isconstant
export blockdiag, eye, rcond
export gdual, ctranspose, inv, ldiv, rdiv
export append, series, parallel, horzcat, vertcat
export order, evalfr, dcgain, opnorm
export gpole, gzero, gpoleinfo, gzeroinfo, gnrank, isregular, isproper, isstable, 
       glinfnorm, ghinfnorm, gl2norm, gh2norm, ghanorm
export gsdec, grnull, glnull, grmcover1, grmcover2, glmcover1, glmcover2
export grcf, glcf, grcfid, glcfid, giofac, goifac
# export grsol, glsol
# export PencilStateSpace, pss, pssdata


abstract type AbstractDynamicalSystem end
abstract type AbstractLTISystem <: AbstractDynamicalSystem end
abstract type AbstractGeneralizedLTIStateSpace <: AbstractLTISystem end
abstract type AbstractDescriptorStateSpace <: AbstractLTISystem end
abstract type AbstractPencilStateSpace <: AbstractLTISystem end
abstract type AbstractRationalTransferFunction <: AbstractLTISystem end

include("types/DescriptorStateSpace.jl")
include("types/RationalFunction.jl")
#include("types/PencilStateSpace.jl")
include("dss.jl")
include("connections.jl")
#include("polynomial_concatenations.jl")
include("rational_concatenations.jl")
include("operations.jl")
include("order_reduction.jl")
include("analysis.jl")
include("decompositions.jl")
include("factorizations.jl")
include("covers.jl")
# include("linsol.jl")
include("nullrange.jl")
include("dstools.jl")
include("dsutils.jl")
end
