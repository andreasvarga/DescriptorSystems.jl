module DescriptorSystems
# Release V0.1

using LinearAlgebra
using MatrixEquations
using MatrixPencils
using Polynomials
using Random

import LinearAlgebra: BlasFloat, BlasReal, BlasComplex, copy_oftype, transpose, adjoint
import Base: +, -, *, /, \, (==), (!=), isapprox, iszero, convert, promote_op, size, length, ndims, 
             hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show
import MatrixPencils: isregular

export DescriptorStateSpace, AbstractDescriptorStateSpace, dss, dssdata, rdss, rss, iszero, order
export gminreal, gir, gbalmr, gsvselect
export blockdiag, eye, rcond
export gdual, ctranspose, inv, ldiv, rdiv
export append, series, parallel, horzcat, vertcat
export order, evalfr, dcgain
# export gpole, gzero, gpolestruct, gzerostruct, gnrank, isregular, isproper, isstable, 
#        gl2norm, gh2norm, ghanorm
# export gsdec
# export grcf, glcf, grcfid, glcfid, giofac, goifac
# export grmcover1, grmcover2, glmcover1, glmcover2
# export grsol, glsol, grnull, glnull
# export PencilStateSpace, pss, pssdata


abstract type AbstractDynamicalSystem end
abstract type AbstractLTISystem <: AbstractDynamicalSystem end
abstract type AbstractGeneralizedLTIStateSpace <: AbstractLTISystem end
abstract type AbstractDescriptorStateSpace <: AbstractLTISystem end
abstract type AbstractPencilStateSpace <: AbstractLTISystem end

include("types/DescriptorStateSpace.jl")
#include("types/PencilStateSpace.jl")
include("dss.jl")
include("connections.jl")
include("operations.jl")
include("order_reduction.jl")
# include("analysis.jl")
# include("factorizations.jl")
# include("covers.jl")
# include("linsol.jl")
# include("nullrange.jl")
include("dstools.jl")
include("dsutils.jl")
end
