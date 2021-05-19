module DescriptorSystems
# Release V0.7*

using LinearAlgebra
using MatrixEquations
using MatrixPencils
using Polynomials
using Random

import LinearAlgebra: BlasFloat, BlasReal, BlasComplex, copy_oftype, transpose, adjoint, opnorm, normalize, rdiv!
import Base: +, -, *, /, \, (==), (!=), ^, isapprox, iszero, convert, promote_op, size, length, ndims, 
             hcat, vcat, hvcat, inv, show, lastindex, require_one_based_indexing, print, show, one, zero, eltype
import MatrixPencils: isregular, rmeval
import Polynomials: AbstractRationalFunction, AbstractPolynomial, poles, isconstant, variable, degree, pqs

export DescriptorStateSpace, AbstractDescriptorStateSpace, dss, dssdata, rdss, rss, iszero, order
export RationalTransferFunction, rtf
export gminreal, gir, gir_lrtran, gbalmr, gsvselect, gss2ss
export gbilin, c2d, dss2rm, dss2pm, timeresp
export confmap, rmconfmap, simplify, normalize, poles, gain, zpk, rtfbilin, numpoly, denpoly, isconstant, sampling_time
export blockdiag, eye, rcond
export gdual, ctranspose, inv, ldiv, rdiv, ginv
export append, series, parallel, horzcat, vertcat
export order, evalfr, dcgain, opnorm, freqresp
export gpole, gzero, gpoleinfo, gzeroinfo, gnrank, isregular, isproper, isstable, 
       glinfnorm, ghinfnorm, gl2norm, gh2norm, ghanorm, gnugap
export gsdec, grnull, glnull, grange, gcrange, grsol, glsol, grmcover1, grmcover2, glmcover1, glmcover2
export grcf, glcf, grcfid, glcfid, gnrcf, gnlcf, giofac, goifac, grsfg, glsfg
export gnehari, glinfldp, glasol, grasol
# export PencilStateSpace, pss, pssdata


abstract type AbstractDynamicalSystem end
abstract type AbstractLTISystem <: AbstractDynamicalSystem end


include("types/DescriptorStateSpace.jl")
include("types/RationalTransferFunction.jl")
#include("types/PencilStateSpace.jl")
include("ginv.jl")
include("dss.jl")
include("connections.jl")
include("operations.jl")
include("conversions.jl")
include("order_reduction.jl")
include("analysis.jl")
include("decompositions.jl")
include("factorizations.jl")
include("covers.jl")
include("linsol.jl")
include("nullrange.jl")
include("model_matching.jl")
include("dstools.jl")
include("dsutils.jl")
end
