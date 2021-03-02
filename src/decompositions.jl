"""
    gsdec(sys; job, smarg, fast,  atol,  atol1, atol2, rtol) -> (sys1, sys2)

Compute for the descriptor system `sys = (A-λE,B,C,D)` with the transfer function matrix
`G(λ)`, the additive spectral decomposition `G(λ) = G1(λ) + G2(λ)` such that `G1(λ)`, 
the transfer function matrix of the descriptor system `sys1 = (A1-λE1,B1,C1,D1)`, 
has only poles in a certain domain of interest `Cg` of the complex plane and `G2(λ)`, 
the transfer function matrix of the descriptor system `sys2 = (A2-λE2,B2,C2,0)`, has
only poles outside of `Cg`. 

The keyword argument `smarg`, if provided, specifies the stability margin for the
stable eigenvalues of `A-λE`, such that, in the continuous-time case, 
the stable eigenvalues have real parts less than or equal to `smarg`, and
in the discrete-time case, the stable eigenvalues
have moduli less than or equal to `smarg`. If `smarg = missing`, the used default values 
are: `smarg = -sqrt(epsm)`, for a continuous-time system, and `smarg = 1-sqrt(epsm)`, 
for a discrete-time system), where `epsm` is the machine precision of the working accuracy. 

The keyword argument `job`, in conjunction with `smarg`, defines the domain of 
interest `Cg`, as follows:

for `job = "finite"`, `Cg` is the whole complex plane without the point at infinity, and 
   `sys1` has only finite poles and `sys2` has only infinite poles (default); 
   the resulting `A2` is nonsingular and upper triangular, while the
   resulting `E2` is nilpotent and upper triangular;   

for `job = "infinite"`, `Cg` is the point at infinity, and 
   `sys1` has only infinite poles and `sys2` has only finite poles and 
   is the strictly proper part of `sys`; 
   the resulting `A1` is nonsingular and upper triangular, while the
   resulting `E1` is nilpotent and upper triangular;   

for `job = "stable"`, `Cg` is the stability domain of eigenvalues defined by `smarg`, and  
    `sys1` has only stable poles and `sys2` has only unstable and infinite poles;    
    the resulting pairs `(A1,E1)` and `(A2,E2)` are in generalized Schur form with
    `E1` upper triangular and nonsingular and `E2` upper triangular;   
 
for `job = "unstable"`,`Cg` is the complement of the stability domain of the 
    eigenvalues defined by `smarg`, and  
    `sys1` has only unstable and infinite poles and `sys2` has only stable poles;    
    the resulting pairs `(A1,E1)` and `(A2,E2)` are in generalized Schur form with
    `E1` upper triangular and `E2` upper triangular  and nonsingular.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. The keyword argument `atol` can be used 
to simultaneously set `atol1 = atol`, `atol2 = atol`. 

The separation of the finite and infinite eigenvalues is performed using 
rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
"""
function gsdec(SYS::DescriptorStateSpace{T}; job::String = "finite", smarg::Union{Real,Missing} = missing, 
               fast::Bool = true,  atol::Real = zero(real(T)),  atol1::Real = atol, atol2::Real = atol, 
               rtol::Real = (SYS.nx*eps(real(float(one(T)))))*iszero(min(atol1,atol2))) where T
    disc = !iszero(SYS.Ts)
    if SYS.E == I
       if job == "finite" 
          return SYS, dss(zeros(T,SYS.ny,SYS.nu), Ts = SYS.Ts)
       elseif job == "infinite"
          return dss(SYS.D, Ts = SYS.Ts), 
                 dss(SYS.A, SYS.B, SYS.C, zeros(T,SYS.ny,SYS.nu), Ts = SYS.Ts)
       elseif job == "stable" || job == "unstable"
          stable_unstable = (job == "stable")
          A, B, C, _, _, blkdims, = ssblkdiag(SYS.A, SYS.B, SYS.C; smarg = smarg, disc = disc, stable_unstable = stable_unstable,  
                                              withQ = false, withZ = false)
          n1 = blkdims[1];
          i1 = 1:n1; i2 = n1+1:SYS.nx 
          return dss(A[i1,i1], B[i1,:], C[:,i1], SYS.D, Ts = SYS.Ts), 
                 DescriptorStateSpace{T}(A[i2,i2], I, B[i2,:], C[:,i2], zeros(T,SYS.ny,SYS.nu), SYS.Ts) 
       else
          error("No such job option")
       end 
    else 
        if job == "finite"
           A, E, B, C, _, _, _, blkdims = fiblkdiag(SYS.A, SYS.E, SYS.B, SYS.C; fast = fast, finite_infinite = true, trinv = false, 
                                                    atol1 = atol1, atol2 = atol2, rtol, withQ = false, withZ = false) 
           n1 = blkdims[1];
        elseif job == "infinite"
           A, E, B, C, _, _, _, blkdims = fiblkdiag(SYS.A, SYS.E, SYS.B, SYS.C; fast = fast, finite_infinite = false, trinv = false, 
                                                    atol1 = atol1, atol2 = atol2, rtol, withQ = false, withZ = false) 
           n1 = blkdims[1];
        elseif job == "stable"
           A, E, B, C, _, _, _, blkdims, = gsblkdiag(SYS.A, SYS.E, SYS.B, SYS.C; smarg = smarg, disc = disc, fast = fast, 
                                                      finite_infinite = true, stable_unstable = true, 
                                                      atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false)
           n1 = blkdims[1];
        elseif job == "unstable"
           A, E, B, C, _, _, _, blkdims, = gsblkdiag(SYS.A, SYS.E, SYS.B, SYS.C; smarg = smarg, disc = disc, fast = fast, 
                                                      finite_infinite = false, stable_unstable = false, 
                                                      atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false)
           n1 = blkdims[1]+blkdims[2];
        else
            error("No such job option")
        end
        i1 = 1:n1; i2 = n1+1:SYS.nx 
        return DescriptorStateSpace{T}(A[i1,i1], E[i1,i1], B[i1,:], C[:,i1], SYS.D, SYS.Ts), 
               DescriptorStateSpace{T}(A[i2,i2], E[i2,i2], B[i2,:], C[:,i2], zeros(T,size(SYS.D)...), SYS.Ts) 
    end
end
