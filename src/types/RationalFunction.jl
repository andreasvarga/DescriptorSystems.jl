"""
    r = RationalTransferFunction(num::AbstractPolynomial, den::AbstractPolynomial, Ts:Real)
    
Construct a rational transfer function model `r` from its numerator and denominator polynomials `num` and `den`, respectively,
and a sampling time `Ts`. 

If `r::RationalTransferFunction{T,λ,P <: Polynomial(T,λ),Ts}` is a rational transfer function system model 
object defined as `r(λ) = num(λ)/den(λ)`, where  `num(λ)` and `den(λ)` are polynomials with coefficients in `T`
and with the indeterminate `λ`, and `Ts` is the sampling time, then:

`r.num` is the numerator polynomial `num(λ)`; 

`r.den` is the denominator polynomial `den(λ)`; 

The sampling time `Ts` can have the following values:

  - `Ts = 0` for a continuous-time system and 
             `λ = s` is the complex variable in the Laplace transform; 

  - `Ts > 0` or `Ts = -1` for a discrete-time system and 
             `λ = z` is the complex variable in the `Z`-transform; 
             `Ts = -1` indicates a discrete-time system with an unspecified sampling time.

The sampling time can be obtained as `r.Ts`.
The symbol (or _variable_) used for the indeterminate `λ` is the common symbol used for the 
indeterminates of the polynomials `num(λ)` and `den(λ)` and can be obtained as `r.var`. 
The roots of the numerator polynomial `num(λ)` (also called _zeros_ of `r(λ)`)
can be obtained as `r.zeros`, while the roots of the denominator polynomial `den(λ)` 
(also called _poles_ of `r(λ)`) can be obtained as `r.poles`. 
The ratio of the leading polynomial coefficients of `num(λ)` and `den(λ)` 
(also called _gain_ of `r(λ)`) can be obtained as `r.gain`.
"""
struct RationalTransferFunction{T,X,P<:AbstractPolynomial{T,X},Ts} <: AbstractRationalFunction{T,X,P}
    ## Define RationalTransferFunction as a subtype of AbstractRationalFunction
    num::P
    den::P
    function RationalTransferFunction{T,X,P,Ts}(num::P, den::P) where{T,X,P<:AbstractPolynomial{T,X}, Ts}
        check_den(den)
        new{T,X,P,Ts}(num,den)
    end
    function RationalTransferFunction{T,X,P,Ts}(num::P, den::P,ts::Union{Real,Nothing}) where{T,X,P<:AbstractPolynomial{T,X}, Ts}
        check_den(den)        
        check_Ts(Ts,ts)        
        new{T,X,P,Ts}(num,den)
    end
    # can promote constants to polynomials too
    function  RationalTransferFunction{T,X,P,Ts}(num::S, den::P, ts::Union{Real,Nothing}) where{S, T,X,P<:AbstractPolynomial{T,X}, Ts}
        check_den(den)
        check_Ts(Ts,ts)        
        new{T,X,P,Ts}(convert(P,num),den)
    end
    function  RationalTransferFunction{T,X,P,Ts}(num::P,den::S, ts::Union{Real,Nothing}) where{S, T,X,P<:AbstractPolynomial{T,X}, Ts}
        check_den(den)
        check_Ts(Ts,ts)
        new{T,X,P,Ts}(num, convert(P,den))
    end
    function RationalTransferFunction{T,X,P}(num::P, den::P, Ts::Union{Real,Nothing}) where{T,X,P<:AbstractPolynomial{T,X}}
        check_den(den)
        Ts′ = standardize_Ts(Ts)
        #new{T,X,P,Val(Ts′)}(num,den)
        new{T,X,P,Ts′}(num,den)
    end
end

#Base.eltype(sys::RationalTransferFunction) = eltype(sys.num)
_eltype(R::RationalTransferFunction) = eltype(R.num)
_eltype(R::VecOrMat{<:RationalTransferFunction}) = length(R) == 0 ? eltype(eltype(Polynomials.coeffs.(numpoly.(R)))) : eltype(eltype(R[1]))
_eltype(R::VecOrMat{<:Polynomial}) = length(R) == 0 ? eltype(eltype(Polynomials.coeffs.(R))) : eltype(eltype(R[1]))

#(pq::RationalTransferFunction)(x) = Polynomials.eval_rationalfunction(x, pq)
(pq::RationalTransferFunction)(x) = pq.num(x)/pq.den(x)

function Base.getproperty(F::RationalTransferFunction{T,X,P,Ts}, d::Symbol)  where {T,X,P,Ts} 
    if d === :Ts
        return Ts
    elseif d === :var
        return X
    elseif d === :zeros
        return roots(getfield(F, :num))
    elseif d === :poles
        return roots(getfield(F, :den))
    elseif d === :gain
        return last(getfield(F, :num)) / last(getfield(F, :den))
    else
        getfield(F, d)
    end
end
Base.propertynames(F::RationalTransferFunction) =
    (:Ts, :var, :zeros, :poles, :gain, fieldnames(typeof(F))...)
poles(F::RationalTransferFunction) = F.poles
poles(F::Polynomial{T}) where T = zeros(T,0)
gain(F::RationalTransferFunction) = F.gain
gain(F::Polynomial) = last(F.coeffs)
gpole(F::RationalTransferFunction) = [F.poles; Inf*ones(Int,max(0,degree(F.num)-degree(F.den)))]
gzero(F::RationalTransferFunction) = [F.zeros; Inf*ones(Int,max(0,degree(F.den)-degree(F.num)))]


function Base.convert(::Type{PQ}, pq::RationalTransferFunction) where {PQ <:RationalFunction}
    p,q = pq
    p//q
end

# alternate constructor
function RationalTransferFunction(p′::P, q′::Q, Ts::Union{Real,Nothing}) where {T,X,P<:AbstractPolynomial{T,X},
                                                                                S,  Q<:AbstractPolynomial{S,X}}

    p,q = promote(p′, q′)
    R = eltype(p)
    RationalTransferFunction{R,X,typeof(p)}(p,q,Ts)
end


function Polynomials.rational_function(::Type{PQ}, p::P, q::Q) where {PQ <:RationalTransferFunction,
                                                          T,X,   P<:AbstractPolynomial{T,X},
                                                          S,   Q<:AbstractPolynomial{S,X}}
    RationalTransferFunction(promote(p,q)..., sampling_time(PQ))
end



## helpers for constructors
# standardize Ts or throw error
function standardize_Ts(Ts)
    isnothing(Ts) || Ts >= 0 || Ts == -1 || 
        throw(ArgumentError("Ts must be either a positive number, 0 (continuous system), or -1 (unspecified)"))
    Ts′ = isnothing(Ts) ? Ts : Float64(Ts)
end
function check_Ts(Ts, ts)
    # ValT(Ts) == promote_Ts(ValT(Ts), ts) || throw(ArgumentError("sampling times have mismatch"))
    ValT(Val(Ts)) == promote_Ts(ValT(Val(Ts)), ts) || throw(ArgumentError("sampling times have mismatch"))
end
function check_den(den)
    iszero(den) && throw(ArgumentError("Cannot create a rational function with zero denominator"))
end

ValT(::Val{T}) where {T} = T
# sampling_time(pq::RationalTransferFunction{T,X,P,Ts}) where {T,X,P,Ts} = ValT(Ts)
# sampling_time(::Type{𝑷}) where {T,X,P,Ts, 𝑷<:RationalTransferFunction{T,X,P,Ts}} = ValT(Ts)
sampling_time(pq::RationalTransferFunction{T,X,P,Ts}) where {T,X,P,Ts} = ValT(Val(Ts))
sampling_time(::Type{𝑷}) where {T,X,P,Ts, 𝑷<:RationalTransferFunction{T,X,P,Ts}} = ValT(Val(Ts))
sampling_time(::Type{<:Number}) = nothing
sampling_time(p::Number) = nothing

"""
    r = rtf(num, den; Ts = rts, var = rvar ) 

Create the rational transfer function `r(λ) = num(λ)/den(λ)` with the polynomials `num(λ)` and `den(λ)`, 
sampling time `rts` and variable name `rvar`, representing 
the transfer function of a single-input single-output system of the form
    
    Y(λ) = r(λ) U(λ),

where `U(λ)` and `Y(λ)` are the Laplace or `Z` transformed input `u(t)` and output `y(t)`, respectively, 
and `λ = s`, the complex variable in the Laplace transform, if `rts = 0`, or  `λ = z`,  
the complex variable in the `Z` transform, if `rts ≠ 0`. 
Both `num` and `den` can be real or complex numbers as well. 

The resulting `r` is such that `r.Ts = rts` (default `rts = 0`) and `r.var = rvar`.
The default value of `rvar` is `rvar = Polynomials.indeterminate(num)` if `num` is a polynomial, 
`rvar = Polynomials.indeterminate(den)` if `num` is a number and `den` is a polynomial, and
`rvar = :s` if both `num` and `den` are numbers.
"""
function rtf(num::Polynomial, den::Polynomial; Ts::Real = 0, 
             var::Symbol = promote_var(num,den)) 
    T = promote_type(eltype(num),eltype(den))
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(num.coeffs), Polynomial{T,var}(den.coeffs))
end
function rtf(num::Polynomial, den::Number; Ts::Real = 0,  var::Symbol = Polynomials.indeterminate(num))
    T = promote_type(eltype(num),eltype(den))
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(num.coeffs), Polynomial{T,var}(den))
end
function rtf(num::Number, den::Polynomial; Ts::Real = 0,  var::Symbol = Polynomials.indeterminate(den))
    T = promote_type(eltype(den),eltype(num))
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(num),Polynomial{T,var}(den.coeffs))
end
function rtf(num::Number, den::Number; Ts::Real = 0, var::Symbol = :s) 
    T = promote_type(eltype(num),eltype(den))
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(num),Polynomial{T,var}(den))
end

function promote_var(num::Polynomial, den::Polynomial)
    m = length(num)
    n = length(den)
    if m > 1 &&  n > 1 
       var = Polynomials.indeterminate(num)
       var != Polynomials.indeterminate(den) && error("The numerator and denominator polynomials must have the same variable")
    elseif m > 1
       var = Polynomials.indeterminate(num)
    elseif  n > 1
       var = Polynomials.indeterminate(den)
    else
       var = :s   # use the default variable for continuous-time systems
    end
    return var
end

"""
    r = rtf(f; Ts = rts, var = rvar) 

Create the rational transfer function `r(λ) = f(λ)` with sampling time `rts` and variable name `rvar`
such that: 

(1) if `f(λ)` is a rational transfer function, then `r.Ts = rts` (default `rts = f.Ts`) and `r.var = rvar` (default `rvar = f.var`);

(2) if `f(λ)` is a rational function, then `r.Ts = rts` (default `rts = 0`) and `r.var = rvar` (default `rvar = Polynomials.indeterminate(f.num)`);

(3) if `f(λ)` is a polynomial, then `r.Ts = rts` (default `rts = 0`) and `r.var = rvar` (default `rvar = Polynomials.indeterminate(f)`);

(4) if `f(λ)` is a ral or complex number, then `r.Ts = rts` (default `rts = 0`) and `r.var = rvar` (default `rvar = :s`);
"""
function rtf(f::RationalTransferFunction; Ts::Real = f.Ts, var::Symbol = f.var)
    T = eltype(eltype(f))
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(f.num.coeffs), Polynomial{T,var}(f.den.coeffs))
end
function rtf(f::AbstractRationalFunction; Ts::Real = 0, var::Symbol = Polynomials.indeterminate(f.num))
    T = eltype(eltype(f))
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(f.num.coeffs), Polynomial{T,var}(f.den.coeffs))
end
function rtf(p::P; Ts::Real = 0, var::Symbol = Polynomials.indeterminate(p)) where {T,X,P<:AbstractPolynomial{T,X}}
    #T = eltype(p)
    #return RationalTransferFunction{T}(Polynomial{T}(p.coeffs,var), Polynomial{T}(one(T),var), Ts)
    #return RationalTransferFunction{T,var}(Polynomial{T,var}(p.coeffs), Polynomial{T,var}(one(T)), Ts)
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(p.coeffs), Polynomial{T,var}(one(T)))
end
function rtf(num::Number; Ts::Real = 0, var::Symbol = :s) 
    T = eltype(num)
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(num), Polynomial{T,var}(one(T)))
end

"""
    r = rtf(var; Ts = rts)
    r = rtf('s') or r = rtf('z'; Ts = rts) 
    r = rtf("s") or r = rtf("z"; Ts = rts) 
    r = rtf(:s) or r = rtf(:z; Ts = rts) 

Create the rational transfer function `r(λ) = λ`, such that `r.var` and `r.Ts` are set as follows:

    (1) `r.var = :s` and `r.Ts = 0` if `var = 's'`, or `var = "s"` or `var = :s` ;

    (2) `r.var = :z` and `r.Ts = rts` if `var = 'z'`, or `var = "z"` or `var = :z`;
    
    (3) `r.var = var` and `r.Ts = rts` (default `rts = 0.`) otherwise.  
"""
function rtf(::Type{T}, var::Union{AbstractString,Char,Symbol}; Ts::Union{Real,Missing} = missing) where T
    if var == "s" || var == 's' || var === :s
       ismissing(Ts) && (Ts = 0)
       Ts == 0. ? (return rtf(Polynomial{T}([0,1],:s),Ts=Ts)) : error("Ts must be zero for a continuous-time system")
    elseif var == "z" || var == 'z' || var === :z
       ismissing(Ts) && (Ts = -1)
       (Ts == -1 || Ts > 0) ? (return rtf(Polynomial{T}([0,1],:z),Ts = Ts)) : error("Ts must be either a positive number or -1 (unspecified)")
    else
       ismissing(Ts) && (Ts = 0.)
       Ts >= 0 || Ts == -1 || error("Ts must be either a positive number, 0 (continuous system), or -1 (unspecified)") 
       return rtf(Polynomial{T}([0,1],var),Ts = Ts) 
    end
end
rtf(var::Union{AbstractString,Char,Symbol}; Ts::Union{Real,Missing} = missing) = rtf(Int, var; Ts = Ts)
function Base.:/(p::AbstractPolynomial,q::AbstractPolynomial)
    RationalFunction(p,q)
end

"""
    r = rtf(z, p, k; Ts = rts, var = rvar) 

Create from the roots (zeros) `z`, poles `p`, gain `k`, sampling time `rts` and variable name `rvar`
the rational transfer function `r(λ) = k*num(λ)/den(λ)`, where `num(λ)` and `den(λ)` are monic polynomials
with roots equal `z` and `p`, respectively, and such that `r.Ts = rts` (default `rts = 0`)
 and `r.var = rvar` (default `rvar = :s` if `Ts = 0` or `rvar = :z` if `Ts ≠ 0`). 
"""
function rtf(zer::Vector{<:Number}, pol::Vector{<:Number}, k::Number; Ts::Real = 0,
             var::Symbol = (Ts == 0 ? :s : :z) )
    Ts >= 0. || Ts == -1. || error("Ts must be either a positive number, 0 (continuous system), or -1 (unspecified)") 
    T = promote_type(eltype(zer),eltype(pol),eltype(k))
    reald = false   # check data correspond to real numerator and denominator
    if eltype(k) <: Real 
       tempc = zer[imag.(zer) .> 0]
       if isempty(tempc) 
          realz = true 
       else
          tempc1 = conj(zer[imag.(zer) .< 0])
          realz = isequal(tempc[sortperm(real(tempc))],tempc1[sortperm(real(tempc1))])
       end
       if realz
          tempc = pol[imag.(pol) .> 0]
          if isempty(tempc) 
             realp = true 
          else
             tempc1 = conj(pol[imag.(pol) .< 0])
             realp = isequal(tempc[sortperm(real(tempc))],tempc1[sortperm(real(tempc1))])
          end
          reald = realp
       end
    end
    num = reald ? real(k*fromroots(zer)) :  k*fromroots(zer)
    den = reald ? real(fromroots(pol)) :  k*fromroots(pol)
    return RationalTransferFunction{T,var,Polynomial{T,var},Float64(Ts)}(Polynomial{T,var}(num.coeffs), Polynomial{T,var}(den.coeffs))
end


## ----
function Base.convert(::Type{PQ}, p::R) where {T,X,P,Ts,PQ <: RationalTransferFunction{T,X,P,Ts}, R<:AbstractRationalFunction}
    PQ(Polynomial(T.(p.num.coeffs),X), Polynomial(T.(p.den.coeffs),X), sampling_time(PQ))
end

function Base.convert(::Type{PQ}, p::P1) where {T,X,P,Ts,PQ <: RationalTransferFunction{T,X,P,Ts}, P1<:AbstractPolynomial}
    PQ(p, one(p), sampling_time(PQ))
end
function Base.convert(::Type{PQ}, p::Number) where {PQ <: RationalTransferFunction}
    PQ(p, one(eltype(PQ)), sampling_time(PQ))
end

function promote_Ts(p,q)
    Ts1,Ts2 = sampling_time.((p,q))
    promote_Ts(Ts1, Ts2)
end

function promote_Ts(Ts1::Union{Real,Nothing}, Ts2::Union{Real,Nothing})
    isnothing(Ts1) && (return Ts2)
    isnothing(Ts2) && (return Ts1)
    Ts1 == Ts2 && (return Ts1)  
    Ts1 == -1 && (Ts2 > 0 ? (return Ts2) : error("Sampling time mismatch"))
    Ts2 == -1 && (Ts1 > 0 ? (return Ts1) : error("Sampling time mismatch"))
    error("Sampling time mismatch")
end

function Base.promote_rule(::Type{PQ}, ::Type{PQ′}) where {T,X,P,Ts,PQ <: RationalTransferFunction{T,X,P,Ts},
                                                           T′,X′,P′,Ts′,PQ′ <: RationalTransferFunction{T′,X′,P′,Ts′}}
    S = promote_type(T,T′)
    Polynomials.assert_same_variable(X,X′)
    Y = X
    Q = promote_type(P, P′)
    ts = promote_Ts(PQ, PQ′)
    #RationalTransferFunction{S,Y,Q,Val(ts)}
    RationalTransferFunction{S,Y,Q,ts}
end
function Base.promote_rule(::Type{PQ}, ::Type{PQ′}) where {T,X,P,Ts,PQ <: RationalTransferFunction{T,X,P,Ts},
                                                           T′,X′,P′,PQ′ <: AbstractRationalFunction{T′,X′,P′}}
    S = promote_type(T,T′)
    Polynomials.assert_same_variable(X,X′)
    Y = X
    Q = promote_type(P, P′)
    ts = sampling_time(PQ)
     #RationalTransferFunction{S,Y,Q,Val(ts)}
    RationalTransferFunction{S,Y,Q,ts}
end

function Base.promote_rule(::Type{PQ}, ::Type{P′}) where {T,X,P,Ts,PQ <: RationalTransferFunction{T,X,P,Ts},
                                                          T′,X′,P′ <: Polynomial{T′,X′}}
    S = promote_type(T,T′)
    Polynomials.assert_same_variable(X,X′)
    Y = X
    Q = promote_type(P, P′)
    ts = sampling_time(PQ)
    #RationalTransferFunction{S,Y,Q,Val(ts)}
    RationalTransferFunction{S,Y,Q,ts}
end

#Base.promote_rule(::Type{PQ}, ::Type{P}) where {PQ <: RationalTransferFunction, P<:AbstractPolynomial} = PQ
Base.promote_rule(::Type{PQ}, ::Type{P}) where {PQ <: RationalTransferFunction, P<:Number} = PQ

Base.zero(::Type{PQ}) where {T,X,P,Ts,PQ <: RationalTransferFunction{T,X,P,Ts}}  = RationalTransferFunction{T,X,P,Ts}(zero(P), one(P))
Base.zero(f::RationalTransferFunction) = zero(typeof(f))
Base.one(::Type{PQ}) where {T,X,P,Ts,PQ <: RationalTransferFunction{T,X,P,Ts}}  = RationalTransferFunction{T,X,P,Ts}(one(P), one(P))
Base.one(f::RationalTransferFunction) = one(typeof(f))
               
"""
     zpk(r) -> (z, p, k)

Compute the roots (zeros) `z`, poles `p` and gain `k` of the rational transfer function `r(λ)`.
"""
function zpk(F::RationalTransferFunction) 
    return F.zeros, F.poles, F.gain
end
function zpk(F::Polynomial{T}) where T
    return roots(F), zeros(T,0), gain(F)
end
   
"""
     rt = adjoint(r)

Compute the adjoint `rt(λ)` of the rational transfer function `r(λ)` such that for 
`r(λ) = num(λ)/den(λ)` we have:

    (1) `rt(λ) = conj(num(-λ))/conj(num(-λ))`, if `r.Ts = 0`; 

    (2) `rt(λ) = conj(num(1/λ))/conj(num(1/λ))`, if `r.Ts = -1` or `r.Ts > 0`.
"""
function Base.adjoint(f::RationalTransferFunction) 
    if f.Ts == 0
       p1 = copy(conj(f.num.coeffs))
       i1 = 2:2:length(p1)
       p1[i1] = -p1[i1]
       p2 = copy(conj(f.den.coeffs))
       i2 = 2:2:length(p2)
       p2[i2] = -p2[i2]
       return RationalTransferFunction(Polynomial(p1,Polynomials.indeterminate(f.num)), Polynomial(p2,Polynomials.indeterminate(f.den)), f.Ts) 
    else
       m = degree(f.num)+1
       n = degree(f.den)+1
       p1 = reverse(conj(f.num.coeffs[1:m]))
       p2 = reverse(conj(f.den.coeffs[1:n]))
       if m >= n
          return RationalTransferFunction(Polynomial(p1,Polynomials.indeterminate(f.num)), Polynomial([zeros(eltype(p2),m-n); p2],Polynomials.indeterminate(f.den)), f.Ts) 
       else
          return RationalTransferFunction(Polynomial([zeros(eltype(p1),n-m); p1],Polynomials.indeterminate(f.num)), Polynomial(p2,Polynomials.indeterminate(f.den)), f.Ts) 
       end         
    end
end
transpose(f::RationalTransferFunction) = f

denpoly(f::RationalTransferFunction) = f.den
numpoly(f::RationalTransferFunction) = f.num
denpoly(f::Number) = one(Polynomial{eltype(f)})
numpoly(f::Number) = Polynomial{eltype(f)}(f)

"""
    n = order(r)

Determine the order `n` of a rational transfer function `r` as the maximum of degrees of its
numerator and denominator polynomials (`n` is also known as the _McMillan degree_ of `r`).  
"""
function order(r::RationalTransferFunction)
    max(degree(r.num),degree(r.den))
end

# operations
function Base.:+(p::R, q::R) where {T,X,P,Ts,R <: RationalTransferFunction{T,X,P,Ts}}
    p0,p1 = pqs(p)
    q0,q1 = pqs(q)
    Polynomials.rational_function(R, p0*q1 + p1*q0, p1*q1)
end
Base.:+(p::Number, q::RationalTransferFunction) = q + p
function Base.:+(p::R,  q::Number) where {T,X,Q,Ts,R <: RationalTransferFunction{T,X,Q,Ts}}
    Tx = promote_type(T,eltype(q))
    p0,p1 = pqs(p)
    #p + q*one(p)
    Polynomials.rational_function(RationalTransferFunction{Tx,X,Polynomial{Tx,X},Ts}, p0 + p1*q, p1)
end
Base.:+(p::Polynomial, q::RationalTransferFunction) = q + p
#Base.:+(p::RationalTransferFunction, q::RationalTransferFunction) = +(promote(p,q)...)
function Base.:+(p::R, q::P) where {T,X,Q,Ts,R <: RationalTransferFunction{T,X,Q,Ts}, T1, P <: Polynomial{T1,X}}
    Tx = promote_type(T,T1)
    p0,p1 = pqs(p)
    Polynomials.rational_function(RationalTransferFunction{Tx,X,Polynomial{Tx,X},Ts}, p0 + p1*q, p1)
end

function Base.:-(p::R, q::R) where {T,X,P,Ts,R <: RationalTransferFunction{T,X,P,Ts}}
    p0,p1 = pqs(p)
    q0,q1 = pqs(q)
    Polynomials.rational_function(R, p0*q1 - p1*q0, p1*q1)
end

function Base.:*(p::R, q::R) where {T,X,P,Ts,R <: RationalTransferFunction{T,X,P,Ts}}
    p0,p1 = pqs(p)
    q0,q1 = pqs(q)
    Polynomials.rational_function(R, p0*q0, p1*q1)
end

# equality
import Base: ==
function ==(p::RationalTransferFunction{T,X,P,T1}, q::RationalTransferFunction{S,Y,Q,T2}) where {T,X,P,T1,S,Y,Q,T2}
    isconstant(p.num) && isconstant(p.den) && isconstant(q.num) && isconstant(q.den) && p(0) == q(0) && return true
    (X == Y && T1 == T2) || return false
    p₀, p₁ = pqs(p)
    q₀, q₁ = pqs(q)
    p₀ * q₁ == q₀ * p₁ || return false
end
function ==(p::RationalTransferFunction{T,X,P,T1}, q::Polynomial{S,Y}) where {T,X,P,T1,S,Y}
    isconstant(p.num) && isconstant(p.den) && isconstant(q) && p(0) == q(0) && return true
    X == Y || return false
    p₀, p₁ = pqs(p)
    p₀ == p₁ * q || return false
end
==(p::Polynomial{S,Y}, q::RationalTransferFunction{T,X,P,T1}) where {S,Y,T,X,P,T1} = ==(q,p)


function Base.isapprox(pq₁::PQ₁, pq₂::PQ₂,
                       rtol::Real = sqrt(eps(float(real(promote_type(T,S))))),
                       atol::Real = zero(real(promote_type(T,S)))) where 
                                {T,X,P,T1,PQ₁<:RationalTransferFunction{T,X,P,T1},
                                 S,Y,Q,T2,PQ₂<:RationalTransferFunction{S,Y,Q,T2}}

    p₁,q₁ = pqs(pq₁)
    p₂,q₂ = pqs(pq₂)

    return X == Y && T1 == T2 && isapprox(p₁*q₂, q₁*p₂; rtol=rtol, atol=atol)
end
function Base.isapprox(pq₁::PQ₁, q₂::Q₂, 
                       rtol::Real = sqrt(eps(float(real(promote_type(T,S))))),
                       atol::Real = zero(real(promote_type(T,S)))) where 
                        {T,X,P,T1,PQ₁<:RationalTransferFunction{T,X,P,T1}, S,Y,Q₂<:Polynomial{S,Y}}

    p₁,q₁ = pqs(pq₁)
    return X == Y && isapprox(p₁, q₁*q₂; rtol=rtol, atol=atol)
end
Base.isapprox(q₁::Q₁, pq₂::PQ₂, 
              rtol::Real = sqrt(eps(float(real(promote_type(T,S))))),
              atol::Real = zero(real(promote_type(T,S)))) where 
              {T,X,P,T2,PQ₂<:RationalTransferFunction{T,X,P,T2},S,Y,Q₁<:Polynomial{S,Y}} = isapprox(pq₂,q₁)
Base.isapprox(q₁::Number, pq₂::PQ₂, 
              rtol::Real = sqrt(eps(float(real(promote_type(T,eltype(q₁)))))),
              atol::Real = zero(real(promote_type(T,eltype(q₁))))) where 
              {T,X,P,T2,PQ₂<:RationalTransferFunction{T,X,P,T2}} = isapprox(pq₂,q₁*one(pq₂))
Base.isapprox(pq₂::PQ₂, q₁::Number, 
              rtol::Real = sqrt(eps(float(real(promote_type(T,eltype(q₁)))))),
              atol::Real = zero(real(promote_type(T,eltype(q₁))))) where 
              {T,X,P,T2,PQ₂<:RationalTransferFunction{T,X,P,T2}} = isapprox(pq₂,q₁*one(pq₂))



## various functions
"""
     normalize(r)

Normalize the rational transfer function `r(λ)` to have a monic denominator polynomial. 
"""
function normalize(f::RationalTransferFunction) 
    k = last(f.den.coeffs)
    rtf(f.num/k,f.den/k,Ts = f.Ts)
end
"""
    rt = confmap(r, f)

Apply the conformal mapping transformation `λ = f(δ)` to the rational transfer function `r(λ)` 
and return `rt(δ) = r(f(δ))`. The resulting `rt` inherits the sampling time and variable of `f`.
"""
function confmap(pol::Polynomial,f::RationalTransferFunction) 
    # perform Horner's algorithm
    n = length(pol)-1
    s = pol[n]*one(f)
    for i in n-1:-1:0
        s = s*f + pol[i]
    end
    return rtf(s,Ts = f.Ts)
end 
function confmap(r::RationalTransferFunction,f::RationalTransferFunction) 
    m = degree(r.num)
    n = degree(r.den)
    pol = f.den
    n1 = degree(pol)
    return m >= n ? rtf(confmap(r.num,f).num,confmap(r.den,f).num*pol^(m-n),Ts = f.Ts,var = f.var) :
                    rtf(confmap(r.num,f).num*pol^(n-m),confmap(r.den,f).num,Ts = f.Ts,var = f.var) 
end
"""
    Rt = rmconfmap(R, f)

Apply elementwise the conformal mapping transformation `λ = f(δ)` to the rational transfer function matrix `R(λ)` 
and return `Rt(δ) = R(f(δ))`. The resulting elements of `Rt` inherit the sampling time and variable of `f`.
"""
function rmconfmap(R::VecOrMat{<:RationalTransferFunction},f::RationalTransferFunction) 
    nrow = size(R,1)
    ncol = size(R,2)
    T = _eltype(R[1])
    #Rt = similar(R,RationalTransferFunction{_eltype(R),f.var}, nrow, ncol)
    Rt = similar(R,RationalTransferFunction{T,f.var,Polynomial{T,f.var},sampling_time(R[1])}, nrow, ncol)
    for j = 1:ncol
        for i = 1:nrow
            Rt[i,j] = confmap(R[i,j],f)
        end
    end
    return Rt 
end
"""
     simplify(r; atol = 0, rtol = atol)

Simplify the rational transfer function `r(λ)` by cancellation of common divisors of numerator and denominator. 
The keyword arguments `atol` and `rtol` are the absolute and relative tolerances for the nonzero
numerator and denominator coefficients. 
"""
function simplify(r::RationalTransferFunction; atol::Real = 0, rtol::Real=10*eps(float(real(_eltype(r))))) 
    m = degree(r.num)
    (macroexpand == 0 || degree(r.den) == 0) && (return r)
    pnum = r.num.coeffs
    pden = r.den.coeffs
    d, u, w,  = MatrixPencils.polgcdvw(pnum, pden, atol = atol, rtol = rtol, maxnit = 3)
    length(u) > m && (return r) # no cancelation, keep original r
    knum = pnum[end]/d[end]/u[end]
    kden = pden[end]/d[end]/w[end]
    return rtf(Polynomial(u*knum,Polynomials.indeterminate(r.num)), Polynomial(w*kden,Polynomials.indeterminate(r.den)), Ts = r.Ts)
end
"""
    Rval = evalfr(R,val) 

Evaluate the rational transfer function matrix `R(λ)` for `λ = val`. 
"""
function evalfr(R::VecOrMat{<:RationalTransferFunction}, val::Number) 
    return pmeval(numpoly.(R),val) ./ pmeval(denpoly.(R),val)
end
"""
    Rval = evalfr(R; fval = 0) 

Evaluate the rational transfer function matrix `R(λ)` for `λ = val`, where `val = im*fval` 
for a continuous-time system or `val = exp(im*fval*abs(Ts))` for a discrete-time system, 
with `Ts` the system sampling time.  
"""
function evalfr(R::VecOrMat{<:RationalTransferFunction}; fval::Number = 0) 
    length(R) > 0 ? Ts = R[1].Ts : Ts = 0
    val = Ts == 0 ? im*abs(fval) : exp(im*abs(fval*Ts))
    return pmeval(numpoly.(R),val) ./ pmeval(denpoly.(R),val)
end
dcgain(R::VecOrMat{<:RationalTransferFunction}) = _eltype(R) <: Complex ? evalfr(R) : real(evalfr(R))
"""
    rval = evalfr(r,val) 

Evaluate the rational transfer function  `r(λ)` for `λ = val`. 
"""
function evalfr(r::RationalTransferFunction,val::Number) 
    return r.num(val) ./ r.den(val)
end
"""
    rval = evalfr(r; fval = 0) 

Evaluate the rational transfer function  `r(λ)` for `λ = val`, where `val = im*fval` 
for a continuous-time system or `val = exp(im*fval*Ts)` for a discrete-time system, 
with `Ts` the system sampling time.   
"""
function evalfr(r::RationalTransferFunction; fval::Number = 0)
    Ts = r.Ts
    val = Ts == 0 ? im*abs(fval) : exp(im*abs(fval*Ts))
    return r.num(val) ./ r.den(val)
end
dcgain(r::RationalTransferFunction) = _eltype(r) <: Complex ? evalfr(r) : real(evalfr(r))
