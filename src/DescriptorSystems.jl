module DescriptorSystems

import Base: *, +, -, /, \, ^, (==), convert, eltype, hcat, hvcat, inv, isapprox, iszero,
    lastindex, length, ndims, nothing, one, print, promote_op, require_one_based_indexing,
    show, size, vcat, zero
import LinearAlgebra: BlasComplex, BlasFloat, BlasReal, adjoint, copy_oftype, hcat,
    normalize, opnorm, rdiv!, transpose
import MatrixPencils: isregular, rmeval
import Polynomials:
    AbstractPolynomial, AbstractRationalFunction, degree, isconstant, poles, pqs, variable

using LinearAlgebra
using MatrixEquations
using MatrixPencils
using Polynomials
using Random
isdefined(Polynomials,:order) && (import Polynomials: order)

export DescriptorStateSpace, AbstractDescriptorStateSpace, dss, dssdata, rdss, rss, iszero, order
export RationalTransferFunction, rtf
export gminreal, gir, gir_lrtran, gbalmr, dsxvarsel, gss2ss, dss2ss, gprescale!, gprescale, gbalqual, pbalqual
export gbilin, c2d, dss2rm, dss2pm, timeresp, stepresp
export confmap, simplify, normalize, poles, gain, zpk, rtfbilin, numpoly, denpoly, isconstant, sampling_time
export blockdiag, eye, rcond
export gdual, ctranspose, inv, ldiv, rdiv, ginv
export append, series, parallel, horzcat, vertcat
export order, evalfr, dcgain, opnorm, freqresp, chess, dssubset, dszeros, dssubsel, dsdiag, feedback
export gpole, gzero, gpoleinfo, gzeroinfo, gnrank, isregular, isproper, isstable, 
       glinfnorm, ghinfnorm, gl2norm, gh2norm, ghanorm, gnugap
export gsdec, grnull, glnull, grange, gcrange, grsol, glsol, grmcover1, grmcover2, glmcover1, glmcover2
export grcf, glcf, grcfid, glcfid, gnrcf, gnlcf, giofac, goifac, grsfg, glsfg
export gnehari, glinfldp, glasol, grasol
# export PeriodicDiscreteDescriptorStateSpace, psreduc_fast
# export PencilStateSpace, pss, pssdata


abstract type AbstractDynamicalSystem end
abstract type AbstractLTISystem <: AbstractDynamicalSystem end
abstract type AbstractLTVSystem <: AbstractDynamicalSystem end
abstract type AbstractDescriptorStateSpace <: AbstractLTISystem end
abstract type AbstractPeriodicStateSpace <: AbstractLTVSystem end


include("types/DescriptorStateSpace.jl")
include("types/RationalTransferFunction.jl")
#include("types/PeriodicStateSpace.jl")
#include("types/PencilStateSpace.jl")
include("ginv.jl")
include("dss.jl")
include("connections.jl")
include("operations.jl")
include("conversions.jl")
include("order_reduction.jl")
include("analysis.jl")
include("timeresp.jl")
include("decompositions.jl")
include("factorizations.jl")
include("covers.jl")
include("linsol.jl")
include("nullrange.jl")
include("model_matching.jl")
include("dstools.jl")
include("dsutils.jl")
end
