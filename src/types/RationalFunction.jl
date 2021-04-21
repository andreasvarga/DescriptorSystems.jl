""" 
     RationalTransferFunction{T,X}(num::Polynomial{T,X}, den::Polynomial{T,X}, Ts::Real) where T <: Number

Construct a rational transfer function model from its numerator and denominator polynomials `num` and `den`, respectively,
and a sampling time `Ts`. 

If `r::RationalTransferFunction{T,X}` is a rational transfer function system model object 
defined as `r(λ) = num(λ)/den(λ)`, where  `num(λ)` and `den(λ)` are polynomials
in the indeterminate `λ`, then:

`r.num` is the numerator polynomial `num(λ)` with coefficients of type `T` and 
indeterminate `λ`; 

`r.den` is the denominator polynomial `den(λ)` with coefficients of type `T` and 
indeterminate `λ`; 

`r.Ts` is the sampling time `Ts`, where `Ts` can have the following values:

  - `Ts = 0` for a continuous-time system and 
             `λ = s` is the complex variable in the Laplace transform; 

  - `Ts > 0` or `Ts = -1` for a discrete-time system and 
             `λ = z` is the complex variable in the `Z`-transform; 
             `Ts = -1` indicates a discrete-time system with an unspecified sampling time;

The symbol (or _variable_) used for the indeterminate `λ` is the common symbol used for the 
indeterminates of the polynomials `num(λ)` and `den(λ)` and can be obtained as `r.var`. 
The roots of the numerator polynomial `num(λ)` (also called _zeros_ of `r(λ)`)
can be obtained as `r.zeros`, while the roots of the denominator polynomial `den(λ)` 
(also called _poles_ of `r(λ)`) can be obtained as `r.poles`. 
The ratio of the leading polynomial coefficients of `num(λ)` and `den(λ)` 
(also called _gain_ of `r(λ)`) can be obtained as `r.gain`.
"""
struct RationalTransferFunction{T,X} <: AbstractLTISystem
    num::Polynomial{T,X}       # numerator polynomial
    den::Polynomial{T,X}       # denominator polynomial
    Ts::Float64                # sampling time (0. - continuous-time, -1. or > 0. - discrete-time)
    function RationalTransferFunction{T,X}(num::Polynomial{T,X}, den::Polynomial{T,X}, Ts::Real) where T <: Number where X
        length(num) > 1 && length(den) > 1 && Polynomials.indeterminate(num) != Polynomials.indeterminate(den) && 
              error("The numerator and denominator polynomials must have the same variable")
        if all(den == zero(den))
            error("Cannot create a rational function with zero denominator")
        elseif all(num == zero(num))
            # The numerator is zero, make the denominator 1
            den = one(den)
        end
        # Validate sampling time
        Ts >= 0 || Ts == -1 || 
             error("Ts must be either a positive number, 0 (continuous system), or -1 (unspecified)")
        new{T,X}(num, den, Float64(Ts))
    end
end
function RationalTransferFunction(num::Polynomial{T1,X}, den::Polynomial{T2,X}, Ts::Real) where {T1,T2,X}
    T = promote_type(T1,T2)
    RationalTransferFunction{T,X}(convert(Polynomial{T,X}, num), convert(Polynomial{T,X}, den), Ts)
end
function RationalTransferFunction{T}(num::Polynomial{T1,X}, den::Polynomial{T2,X}, Ts::Real) where {T,T1,T2,X}
    RationalTransferFunction{T,X}(convert(Polynomial{T,X}, num), convert(Polynomial{T,X}, den), Ts)
end
function Base.getproperty(F::RationalTransferFunction, d::Symbol)
    if d === :zeros
        return roots(getfield(F, :num))
    elseif d === :poles
        return roots(getfield(F, :den))
    elseif d === :gain
        return last(getfield(F, :num)) / last(getfield(F, :den))
    elseif d === :var
        return Polynomials.indeterminate(getfield(F, :num))
    else
        getfield(F, d)
    end
end
Base.propertynames(F::RationalTransferFunction) =
    (:zeros, :poles, :gain, :var, fieldnames(typeof(F))...)

poles(F::RationalTransferFunction) = F.poles
poles(F::Polynomial{T}) where T = zeros(T,0)
gain(F::RationalTransferFunction) = F.gain
gain(F::Polynomial) = last(F.coeffs)

Base.length(F::RationalTransferFunction) = 1

"""
    r = rtf(num, den; Ts = 0, var = Polynomials.indeterminate(num)) 

Create for the polynomials `num(λ)` and `den(λ)`, sampling time `Ts` and variable name `var`
the rational transfer function `r(λ) = num(λ)/den(λ)` of a single-input single-output system of the form
    
    Y(λ) = r(λ) U(λ),

where `U(λ)` and `Y(λ)` are the Laplace transformed, if `Ts = 0`, or `Z`-transformed, if `Ts ≠ 0`,
system input `u(t)` and system output `y(t)`, respectively, and 
`λ = s`,  if `Ts = 0`, or  `λ = z`,  if `Ts ≠ 0`. 
The resulting `r` is such that `r.Ts = Ts` and `r.var = var`.  

Both `num(λ)` and `den(λ)` can be real or complex numbers as well. 
"""
function rtf(num::Polynomial, den::Polynomial; Ts::Real = 0, 
             var::Symbol = promote_var(num,den)) 
    T = promote_type(eltype(num),eltype(den))
    return RationalTransferFunction{T}(Polynomial{T}(num.coeffs,var), Polynomial{T}(den.coeffs,var), Ts)
end
function rtf(num::Polynomial, den::Number; Ts::Real = 0) 
    T = promote_type(eltype(num),eltype(den))
    return RationalTransferFunction{T}(num, Polynomial(den,Polynomials.indeterminate(num)), Ts)
end
function rtf(num::Number, den::Polynomial; Ts::Real = 0) 
    T = promote_type(eltype(den),eltype(num))
    return RationalTransferFunction{T}(Polynomial(num,Polynomials.indeterminate(den)), den, Ts)
end
function rtf(num::Number, den::Number; Ts::Real = 0, var::Symbol = :s) 
    T = promote_type(eltype(num),eltype(den))
    return  RationalTransferFunction{T}(Polynomial{T}(num,var), Polynomial{T}(den,var), Ts)
end
function rtf(num::Number; Ts::Real = 0, var::Symbol = :s) 
    T = eltype(num)
    return RationalTransferFunction{T}(Polynomial{T}(num,var), Polynomial{T}(one(T),var), Ts)
end
"""
    r = rtf(f; Ts = f.Ts, var = f.var) 

Set for the rational transfer function `r(λ) = f(λ)` the sampling time to `Ts` and variable name to `var`
such that `r(λ) = f(λ)`, `r.Ts = Ts` (default `r.Ts = f.Ts`) and `r.var = var` (default `r.var = f.var`).
"""
function rtf(f::RationalTransferFunction; Ts::Real = f.Ts, var::Symbol = f.var)
    T = eltype(f)
    return RationalTransferFunction{T}(Polynomial{T}(f.num.coeffs,var), Polynomial{T}(f.den.coeffs,var), Ts)
end
"""
    r = rtf(p; Ts = 0, var = Polynomials.indeterminate(p)) 

Create for the polynomial `p(λ)`, sampling time `Ts` and variable `var`
the rational transfer function `r(λ) = p(λ)` such that `r.Ts = Ts` (default `r.Ts = 0`)
and `r.var = var` (default  (default `r.var = Polynomials.indeterminate(p)`).
`p(λ)` can be a real or complex number as well. 
"""
function rtf(p::Polynomial{T,X}; Ts::Real = 0, var::Symbol = Polynomials.indeterminate(p)) where {T,X}
    #T = eltype(p)
    #return RationalTransferFunction{T}(Polynomial{T}(p.coeffs,var), Polynomial{T}(one(T),var), Ts)
    #return RationalTransferFunction{T,var}(Polynomial{T,var}(p.coeffs), Polynomial{T,var}(one(T)), Ts)
    return RationalTransferFunction{T,var}(Polynomial{T,var}(p.coeffs), Polynomial{T,var}(one(T)), Ts)
end
# function rtf(p::Polynomial; Ts::Real = 0, var::Symbol = Polynomials.indeterminate(p))
#     T = eltype(p)
#     #return RationalTransferFunction{T}(Polynomial{T}(p.coeffs,var), Polynomial{T}(one(T),var), Ts)
#     #return RationalTransferFunction{T,var}(Polynomial{T,var}(p.coeffs), Polynomial{T,var}(one(T)), Ts)
#     return RationalTransferFunction{T,var}(p, Polynomial{T,var}(one(T)), Ts)
# end
Base.eltype(sys::RationalTransferFunction) = eltype(sys.num)
_eltype(R::RationalTransferFunction) = eltype(R.num)
_eltype(R::VecOrMat{<:RationalTransferFunction}) = length(R) == 0 ? eltype(eltype(Polynomials.coeffs.(numpoly.(R)))) : eltype(R[1])
_eltype(R::VecOrMat{<:Polynomial}) = length(R) == 0 ? eltype(eltype(Polynomials.coeffs.(R))) : eltype(R[1])
"""
    r = rtf(var; Ts = 0.)
    r = rtf('s'; Ts = 0.) or r = rtf('z'; Ts = -1.) 
    r = rtf("s"; Ts = 0.) or r = rtf("z"; Ts = -1.) 
    r = rtf(:s; Ts = 0.) or r = rtf(:z; Ts = -1.) 

Create the rational transfer function `r(λ) = λ`, such that `r.var` and `r.Ts` are set as follows:
    (1) `r.var = :s` and `r.Ts = 0` if `var = 's'`, or `var = "s"` or `var = :s` ;
    (2) `r.var = :z` and `r.Ts = Ts` if `var = 'z'`, or `var = "z"` or `var = :z`;
    (3) `r.var = var` and `r.Ts = Ts` (default `Ts = 0.`) otherwise.  
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
"""
    r = rtf(z, p, k; Ts = 0, var = :s) 

Create from the roots (zeros) `z`, poles `p`, gain `k`, sampling time `Ts` and variable name `var`
the rational transfer function `r(λ) = k*num(λ)/den(λ)`, where `num(λ)` and `den(λ)` are monic polynomials
with roots equal `z` and `p`, respectively, and such that `r.Ts = Ts` (default `r.Ts = 0`)
 and `r.var = var` (default `r.var = :s` if `Ts = 0` or `r.var = :z` if `Ts ≠ 0`). 
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
    return RationalTransferFunction{T}(Polynomial{T}(num.coeffs,var), Polynomial{T}(den.coeffs,var), Ts)
end


function +(rtf1::RationalTransferFunction{T1,X}, rtf2::RationalTransferFunction{T2,X}) where {T1,T2,X}
    Ts = promote_Ts(rtf1.Ts,rtf2.Ts)
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rtf1.num*rtf2.den + rtf2.num*rtf1.den, rtf1.den*rtf2.den, Ts)
end
function +(rf::RationalTransferFunction{T1,X}, pol::Polynomial{T2,X}) where {T1,T2,X}
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rf.num + pol*rf.den, rf.den, rf.Ts)
end
+(pol::Polynomial,rf::RationalTransferFunction) = rf + pol
function +(f::RationalTransferFunction, n::Number) 
    T1 = promote_type(eltype(f), eltype(n))
    return RationalTransferFunction{T1}(f.num + n*f.den, f.den, f.Ts)
end
+(n::Number, f::RationalTransferFunction) = f + n
-(f::RationalTransferFunction) = RationalTransferFunction{eltype(f)}(-f.num, f.den, f.Ts)
function -(rtf1::RationalTransferFunction{T1,X}, rtf2::RationalTransferFunction{T2,X}) where {T1,T2,X}
    Ts = promote_Ts(rtf1.Ts,rtf2.Ts)
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rtf1.num*rtf2.den - rtf2.num*rtf1.den, rtf1.den*rtf2.den, Ts)
end
function -(rf::RationalTransferFunction{T1,X}, pol::Polynomial{T2,X}) where {T1,T2,X}
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rf.num - pol*rf.den, rf.den, rf.Ts)
end
-(pol::Polynomial,rf::RationalTransferFunction) = +(pol,-rf)
function -( n::Number, f::RationalTransferFunction) 
    T = promote_type(eltype(f), eltype(n))
    return RationalTransferFunction{T}(n*f.den - f.num, f.den, f.Ts)
end
-(f::RationalTransferFunction, n::Number) = +(f, -n)
function *(rtf1::RationalTransferFunction{T1,X}, rtf2::RationalTransferFunction{T2,X}) where {T1,T2,X}
    Ts = promote_Ts(rtf1.Ts,rtf2.Ts)
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rtf1.num*rtf2.num, rtf1.den*rtf2.den, Ts)
end
function *(rf::RationalTransferFunction{T1,X}, pol::Polynomial{T2,X}) where {T1,T2,X}
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rf.num * pol, rf.den, rf.Ts)
end
*(pol::Polynomial,rf::RationalTransferFunction) = rf * pol
function *(rf::RationalTransferFunction, n::Number) 
    T = promote_type(_eltype(rf), eltype(n))
    return RationalTransferFunction{T}(rf.num*n, rf.den, rf.Ts)
end
*(n::Number, rf::RationalTransferFunction) = *(rf,n)

function /(rf::RationalTransferFunction{T1,X}, pol::Polynomial{T2,X}) where {T1,T2,X}
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rf.num, rf.den*pol, rf.Ts)
end
function /(pol::Polynomial{T1,X},rf::RationalTransferFunction{T2,X}) where {T1,T2,X}
    T = promote_type(T1, T2)
    return RationalTransferFunction{T}(rf.den*pol, rf.num, rf.Ts)
end
function /(p1::Polynomial{T1,X}, p2::Polynomial{T2,X}) where {T1,T2,X}
    T = promote_type(T1, T2)
    var = Polynomials.indeterminate(p1)
    return RationalTransferFunction{T}(Polynomial{T}(p1.coeffs,var), Polynomial{T}(p2.coeffs,var), 0.)
end
function /(n::Number, p::Polynomial)
    T = promote_type(eltype(p), eltype(n))
    var = Polynomials.indeterminate(p)
    return RationalTransferFunction{T}(Polynomial{T}(T(n),var), Polynomial{T}(p.coeffs,var), 0.)
end

function /(n::Number, f::RationalTransferFunction) 
    T = promote_type(_eltype(f), eltype(n))
    return RationalTransferFunction{T}(n*f.den, f.num, f.Ts)
end
/(f::RationalTransferFunction, n::Number) = f*(1/n)
/(f1::RationalTransferFunction, f2::RationalTransferFunction) = f1*(1/f2)

function ==(f1::RationalTransferFunction, f2::RationalTransferFunction)
    f1.num * f2.den == f2.num * f1.den || (return false) 
    ( isconstant(f1) || isconstant(f2) ) && (return true)
    return (f1.Ts == f2.Ts && f1.var == f2.var) 
end
function ==(f1::RationalTransferFunction, f2::Polynomial)
    f1.num  == f2 * f1.den || (return false)
    ( isconstant(f1) || isconstant(f2) ) && (return true)
    return  f1.var == Polynomials.indeterminate(f2)
end
==(f1::Polynomial, f2::RationalTransferFunction) = ==(f2,f1)
==(f::RationalTransferFunction, n::Number) = (f.num  == n * f.den ) 
==(n::Number, f::RationalTransferFunction) = (f.num  == n * f.den ) 

function isconstant(f::RationalTransferFunction)
    return degree(f.num) <= 0 && degree(f.den) == 0
end
function isconstant(f::Polynomial)
    return degree(f) <= 0 
end
isconstant(f::Number) = true
variable(f::RationalTransferFunction) = variable(f.num)


function isapprox(r1::RationalTransferFunction, r2::RationalTransferFunction;
                  rtol::Real = sqrt(eps(float(real(promote_type(_eltype(r1),_eltype(r2)))))), atol::Real = 0) 
  (r1.Ts == r2.Ts && r1.var == r2.var) || (return false)
  p1 = r1.num * r2.den
  p2 = r1.den * r2.num
  isapprox(coeffs(p1), coeffs(p2); rtol = rtol, atol = atol) 
end
function isapprox(r1::RationalTransferFunction, r2::Polynomial;
                  rtol::Real = sqrt(eps(float(real(promote_type(_eltype(r1),eltype(r2)))))), atol::Real = 0) 
  p1 = r1.num 
  p2 = r1.den * r2
  isapprox(coeffs(p1), coeffs(p2); rtol = rtol, atol = atol) || (return false)
  ( isconstant(r1) || isconstant(r2) ) && (return true)
  return  r1.var == Polynomials.indeterminate(r2)
end
function isapprox(r1::Polynomial, r2::RationalTransferFunction;
    rtol::Real = sqrt(eps(float(real(promote_type(eltype(r1),_eltype(r2)))))), atol::Real = 0) 
    p1 = r1 * r2.den
    p2 = r2.num
    isapprox(coeffs(p1), coeffs(p2); rtol = rtol, atol = atol) || (return false)
    ( isconstant(r1) || isconstant(r2) ) && (return true)
    return  r2.var == Polynomials.indeterminate(r1)
end
isapprox(f::RationalTransferFunction, n::Number) = isapprox(f.num, n * f.den ) 
isapprox(n::Number, f::RationalTransferFunction) = isapprox(f.num, n * f.den ) 

function ^(f::RationalTransferFunction{T,X}, n::Int) where {T,X}
    return n >= 0 ? RationalTransferFunction{T,X}(f.num^n, f.den^n, f.Ts) : RationalTransferFunction{T,X}(f.den^n, f.num^n, f.Ts) 
end

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
function adjoint(f::RationalTransferFunction) 
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
transpose(R::VecOrMat{<:RationalTransferFunction}) = permutedims(R)
adjoint(R::VecOrMat{<:RationalTransferFunction}) = adjoint.(permutedims(R))
"""
    rinv = inv(r)

Build the inverse `rinv` of a nonzero rational transfer function `r` such that `rinv(λ) = 1/r(λ)`. 
"""
function inv(r::RationalTransferFunction) 
    RationalTransferFunction(r.den, r.num, r.Ts)
end
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
Base.promote_rule(::Type{RationalTransferFunction{T1,X}}, ::Type{T2}) where {T1 <:Number, T2<:Number, X} =
    RationalTransferFunction{promote_type(T1, T2),X}
Base.convert(::Type{RationalTransferFunction{T}}, n::Number) where T = rtf(promote_type(T,eltype(n))(n), Ts = 0.)


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
_zerortf(::Type{RationalTransferFunction{T,X}},Ts::Real = 0,var::Symbol = :s) where {T,X} =
    RationalTransferFunction{T}(Polynomial{T}(zero(T),var), Polynomial{T}(one(T),var), Ts)
_zerortf(::Type{RationalTransferFunction{T}},Ts::Real = 0,var::Symbol = :s) where T =
    RationalTransferFunction{T}(Polynomial{T}(zero(T),var), Polynomial{T}(one(T),var), Ts)
Base.zero(::Type{RationalTransferFunction})  = _zerortf(RationalTransferFunction{Float64},0.,:s)
Base.zero(::Type{RationalTransferFunction{T}}) where T  = _zerortf(RationalTransferFunction{T},0.,:s)
Base.zero(f::RationalTransferFunction) = _zerortf(typeof(f),f.Ts,f.var)
Base.zero(::Type{RationalTransferFunction{T,X}}) where T where X  = _zerortf(RationalTransferFunction{T,X},0.,:s)

_onertf(::Type{RationalTransferFunction{T,X}},Ts::Real = 0,var::Symbol = :y) where {T,X} =
    RationalTransferFunction{T}(Polynomial{T}(one(T),var), Polynomial{T}(one(T),var), Ts)
_onertf(::Type{RationalTransferFunction{T}},Ts::Real = 0,var::Symbol = :s) where T  =
    RationalTransferFunction{T}(Polynomial{T}(one(T),var), Polynomial{T}(one(T),var), Ts)
Base.one(::Type{RationalTransferFunction})  = _onertf(RationalTransferFunction{Float64},0.,:s)
Base.one(::Type{RationalTransferFunction{T}}) where T  = _onertf(RationalTransferFunction{T},0.,:s)
Base.one(f::RationalTransferFunction) = _onertf(typeof(f),f.Ts,f.var)
Base.one(::Type{RationalTransferFunction{T,X}}) where T where X  = _onertf(RationalTransferFunction{T,X},0.,:s)

"""
     normalize(r; atol = 0, rtol = atol)

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
    n = length(pol)
    s = pol[n]*one(f)
    for i in n-1:-1:0
        s = s*f + pol[i]
    end
    return rtf(s.num,1,Ts = f.Ts)
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
    Rt = similar(R,RationalTransferFunction{_eltype(R),f.var}, nrow, ncol)
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
for a continuous-time system or `val = exp(im*fval*Ts)` for a discrete-time system, 
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

