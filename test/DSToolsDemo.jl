module DSToolsDemo

# # DSTOOLSDEMO    Demonstration of the Descriptor System Tools (DSTOOLS).
# #                A. Varga.

using DescriptorSystems
using LinearAlgebra
using Polynomials
using Test

function pause()
    # Comment out the following two lines for a non-interactive execution
    println("Press any key to continue") 
    readline()  
end

s = rtf('s'); z = rtf('z'); # define the complex variables s and z  
println("s = rtf('s'); z = rtf('z'); # define the complex variables s and z ")
pause()

println("Gc = [s^2 s/(s+1); 0 1/s]   # define the 2-by-2 improper Gc(s)")
Gc = [s^2 s/(s+1); 0 1/s]   # define the 2-by-2 improper Gc(s)
Gd = [z^2 z/(z-2); 0 1/z]   # define the 2-by-2 improper Gd(z)
println(Gc)
pause()

println("Gd = [z^2 z/(z-2); 0 1/z]   # define the 2-by-2 improper Gd(z)")
println(Gd)
pause()

println("Build LTI descriptor realizations of Gc(s) and Gd(z) ")
println("sysc = dss(Gc, minimal = true);  # build continuous-time descriptor system realization")
sysc = dss(Gc, minimal = true);  # build continuous-time descriptor system realization
println(sysc)
pause()

println("dss(Gd, minimal = true)  # build discrete-time descriptor system realization")
sysd = dss(Gd, minimal = true);   # build continuous-time descriptor system realization
println(sysd)
pause()

println("Compute poles and zeros of Gc(s)")
println("gpole(sysc)   #  computes all poles (finite and infinite)")
display(gpole(sysc))
pause()

println("gzero(sysc)   #  computes all zeros (finite and infinite)")
display(gzero(sysc))
pause()

println("Descriptor systems (A-λE,B,C,D) with singular pole pencil A-λE are not supported in DSTOOLS!")
println("Non-regular A-λE can be detected in several ways. ")
println("(1) non-regular A-λE can be detected when building a descriptor system ")
println("A = zeros(1,1); E = zeros(1,1); B = ones(1,1); C = ones(1,1); D = zeros(1,1);")
println("dss(A,E,B,C,D,check_reg = true)")
A = zeros(1,1); E = zeros(1,1); B = ones(1,1); C = ones(1,1); D = zeros(1,1);
try
  dss(A,E,B,C,D,check_reg = true)  # check regularity when building the descriptor system
catch err
  println(err)
end
pause()

println("(2) non-regular A-λE can be explicitly detected using isregular")
println("syst = dss(A,E,B,C,D); # no regularity check performed when building a descriptor system")
println("isregular(syst) == false")
syst = dss(A,E,B,C,D);
println(@test isregular(syst) == false) 
pause()

println("(3) non-regular A-λE can be detected using isproper")
println("isproper(syst) == false")
syst = dss(A,E,B,C,D);
println(@test isproper(syst) == false)
pause()

println("(4) gpole can also be used to check regularity of the pole pencil")
println("gpole(syst)    # the system has some poles equal to NaN")
display(gpole(syst))
pause()

println("Let's try some decomposition and factorization functions\n")

println("Use gsdec to compute the separation of proper and polynomial parts of Gc(s)")
println("sysf, sysi = gsdec(sysc); # Gc(s) = Gcp(s) + Gci(s)")
sysf, sysi = gsdec(sysc); 
println("@test iszero(sysc-sysf-sysi)  # checking the decomposition")
println(@test iszero(sysc-sysf-sysi))
pause()
println("dss2rm(sysf)  # display proper part Gcp(s)")
println(dss2rm(sysf))
pause()
println("gpole(sysf)  # display polynomial part Gcp(s)")
println(dss2rm(sysi))
pause()


println("Compute the stable and proper right coprime factorization of Gc(s) as")
println("Gc(s) = N(s)*inv(M(s)), with N(s) and M(s) stable and proper,")
println(" having a stability degree -1 and poles assigned to [-2,-3,-4]")
println("sysn, sysm = grcf(sysc,evals = [-2,-3,-4],sdeg = -1,mindeg = true, mininf = true);")
sysn, sysm = grcf(sysc,evals = [-2,-3,-4], sdeg = -1, mindeg = true,  mininf = true);
println("@test iszero(sysc*sysm-sysn)  # checking the factorization")
println(@test iszero(sysc*sysm-sysn))
pause()

println("@test isproper(sysm) && isproper(sysn) # checking properness of factors")
println(@test isproper(sysm) && isproper(sysn) )   
pause()

println("@test (maximum(real(gpole(sysm))) < 0 && maximum(real(gpole(sysn))) < 0) # checking stability of poles")
println(@test (maximum(real(gpole(sysm))) < 0 && maximum(real(gpole(sysn))) < 0) )   
pause()

println("@test isempty(gzero(gminreal([sysn;sysm]))) # checking coprimeness:  [N(s);M(s)] has no zeros")
println(@test isempty(gzero(gminreal([sysn;sysm]))))
pause()


println("Compute the right coprime factorization with inner denominator of Gd(z) as")
println("Gd(z) = N(z)*inv(M(z)), with N(z) and M(z) stable and proper, and M(z) inner")
println("sysni, sysmi = grcfid(sysd);")
sysni, sysmi = grcfid(sysd);
println("@test iszero(sysc*sysm-sysn)  # checking the factorization")
println(@test iszero(sysc*sysm-sysn))
pause()

println("@test maximum(abs.(gpole(sysmi))) < 1 && maximum(abs.(gpole(sysni))) < 1) # checking stability of poles")
println(@test maximum(abs.(gpole(sysmi))) < 1 && maximum(abs.(gpole(sysni))) < 1  )   
pause()

println("@test isempty(gzero(gminreal([sysn;sysm]))) # checking the innerness: conj(M(z))*M(z)-I = 0")
println(@test iszero(sysmi'*sysmi-I))
pause()

println("@test isempty(gzero(gminreal([sysn;sysm]))) # checking coprimeness:  [N(z);M(z)] has no zeros")
println(@test isempty(gzero(gminreal([sysn;sysm]))))
pause()


println("Compute the inner-outer factorization of stable 3x3 G(s) with rank 2 ")
println("s = Polynomial([0, 1],'s'); ")
s = Polynomial([0, 1],'s'); 
println("num = [(s-1) s 1; 0 (s-2) (s-2); (s-1) (s^2+2*s-2) (2*s-1)]; # numerators")
num = [(s-1) s 1; 0 (s-2) (s-2); (s-1) (s^2+2*s-2) (2*s-1)]; # numerators
println(num)
pause()
println("den = [(s+2) (s+2) (s+2); 1 (s+1)^2 (s+1)^2; (s+2) (s+1)*(s+2) (s+1)*(s+2)]; # denominators")
den = [(s+2) (s+2) (s+2); 1 (s+1)^2 (s+1)^2; (s+2) (s+1)*(s+2) (s+1)*(s+2)]; # denominators
println(den)
pause()
println("sys = dss(num,den,minimal = true, atol = 1.e-7);  # build a minimal descriptor realization")
sys = dss(num,den,minimal = true, atol = 1.e-7); 
println(sys)
pause()
println("Analysis of some properties")
println("gpole(sys);  # the system is stable")
display(gpole(sys))    # the system is stable
pause()
println("gzero(sys,atol1=1.e-7)    # the system has 2 unstable zeros and an infinite zero")
display(gzero(sys,atol1=1.e-7) )   # the system has 2 unstable zeros and an infinite zero
pause()
println("nr = gnrank(sys,atol1=1.e-7)    # the normal rank of G(s) is 2")
nr = gnrank(sys,atol1=1.e-7);  # the normal rank of G(s) is 2
println(nr)
pause()

println("Compute the inner-quasi-outer factorization of G(s) as G(s) = Gi(s)*[Go(s);0],") 
println("with Gi(s) inner and square, and Go(s) quasi-outer (i.e., full row rank, ")
println("and without zeros in the open right-half plane), can be computed by using")
println("sysi, syso, info = giofac(sys)")
sysi, syso, info = giofac(sys);
pause()

println("Checking the factorization: Gi(:,1:nr)(s)*Go(s)-G(s) = 0")
println("@test iszero(sysi[:,1:nr]*syso-sys,atol1=1.e-7)")
println(@test iszero(sysi[:,1:nr]*syso-sys,atol1=1.e-7))
pause()

println("Checking the innerness of Gi(s)")
println("@test iszero(sysi'*sysi-I,atol1=1.e-7)")
println(@test iszero(sysi'*sysi-I,atol1=1.e-7))
pause()


println("If G(s) contains a so-called free inner factor, then this factor")
println("is included in Gi(s), but the realization of Go(s) is not minimal")
println("@test order(syso)-order(gir(syso)) > 0  # a free inner factor is present in G(s)")
println(@test order(syso)-order(gir(syso)) > 0 ) # a free inner factor is present in G(s) 
pause()

println("Checking that Go(s) has no zeros in the open right-half plane")
println("zer = gzero(syso,atol1 = 1.e-7);  # compute zeros of Go")
zer = gzero(syso,atol1 = 1.e-7);
println("@test all(real.(zer[.!isinf.(zer)]) .< 0 )")
println(@test all(real.(zer[.!isinf.(zer)]) .< 0 ))
pause()

##
println("Solution of linear rational equations G(s)*X(s) = F(s)")
println("Consider the Wang and Davison example (IEEE Trans. Autom. Contr.,1973)")
println("to determine a right inverse of G(s) by solving G(s)*X(s) = I.")
println("s = Polynomial([0, 1],'s'); # define polynomial s")
s = Polynomial([0, 1],'s'); 
println("num =  [ s+1 s+3 s^2+3*s; s+2 s^2+2*s 0 ]; # numerators")
num =  [ s+1 s+3 s^2+3*s; s+2 s^2+2*s 0 ]; # numerators
println(num)
pause()
println("den = [s^2+3*s+2 s^2+3*s+2 s^2+3*s+2; s^2+3*s+2 s^2+3*s+2 s^2+3*s+2]; # denominators")
den = [s^2+3*s+2 s^2+3*s+2 s^2+3*s+2; s^2+3*s+2 s^2+3*s+2 s^2+3*s+2]; # denominators
println(den)
pause()
println("sysg = dss(num,den,minimal = true, atol = 1.e-7);  # build a minimal descriptor realization of G(s)")
sysg = dss(num,den,minimal = true, atol = 1.e-7); 
println(sysg)
pause()
println("sysf = dss([1 0;0 1.]);  # build a minimal descriptor realization of F(s) = I")
sysf = dss([1 0;0 1.]);
println(sysf)
pause()

println("zer = gzero(sysg)   # G(s) has no zeros, thus stable right inverses exist")
zer = gzero(sysg)         # G(s) has no zeros, thus stable right inverses exist
println("zer = $zer")
pause()

println("Compute a stable right inverse with poles in [-1 -2 -3] by using")
println("sysx, info = grsol(sysg,sysf,poles = [-1, -2, -3]);")
sysx, info = grsol(sysg, sysf, poles = [-1, -2, -3]); 
println(sysx)
pause() 

println("Checking the solution: G(s)*X(s) - I = 0")
println("iszero(sysg*sysx-I,atol=1.e-7) == 0")
println(@test iszero(sysg*sysx-I,atol=1.e-7))   #  G(s)*X(s) - I = 0
pause 
          
println("Check assigned poles: gpole(sysx) = [-1, -2, -3]")
println(@test sort(gpole(sysx)) ≈ sort([-1, -2, -3]))
pause() 

println("We can also compute a right inverse of least order = 2")
println("sysxmin, = grsol(sysg,sysf,mindeg = true); order(sysxmin) == 2")
sysxmin, = grsol(sysg,sysf,mindeg = true); 
println(@test iszero(sysg*sysxmin-I,atol=1.e-7) && order(sysxmin) == 2)
pause() # Press any key to continue ...

gpole(sysxmin)            
println("The least order inverse is unstable: max(real(gpole(sysxmin))) > 0")
println(@test maximum(real(gpole(sysxmin))) > 0)
pause() 

##
println("\nWe can try to explicitly determine a least order right inverse")
println("using stable generators (X0(s),XN(s)), where X0(s) is a particular")
println("stable right inverse satisfying G(s)*X0(s) = I and XN(s) is a stable")
println("basis of the right nullspace of G(s), satisfying G(s)*XN(s) = 0;")
println("all solutions of G(s)*X(s) = I are given by X(s) = X0(s)+XN(s)*Y(s),")
println("where Y(s) is arbitrary\n")

println("Determine the generators X0 and XN with poles assigned to [-1 -2 -3]")
println("_,_,sysgen = grsol(sysg,sysf,poles = [-1, -2, -3],solgen = true);")
_,_,sysgen = grsol(sysg,sysf,poles = [-1, -2, -3],solgen = true);
pause()

println("Compute a least order solution X2(s) = X0(s)+XN(s)*Y2(s) using") 
println("order reduction based on minimal dynamic covers of Type 2")

println("sysx2,sysy2, = grmcover2(sysgen,2); order(gminreal(sysx2)) == 2")
sysx2,sysy2, = grmcover2(sysgen,2); 
println(@test order(gminreal(sysx2)) == 2)
pause() 

println("Checking the solution: iszero(sysg*sysx2-I,atol=1.e-7) == true")
println(@test iszero(sysg*sysx2-I,atol=1.e-7) )  #  G(s)*X(s) - I = 0
pause() 

println("The least order inverse is unstable: max(real(gpole(sysx2))) > 0")
println(@test maximum(real(gpole(sysx2))) > 0)
pause() 

println("Checking the minimal cover reduction results: X2(s) = X0(s)+XN(s)*Y2(s)")
println("iszero(sysx2-sysgen[:,1:2]-sysgen[:,3]*sysy2,atol=1.e-7) == true")
println(@test iszero(sysx2-sysgen[:,1:2]-sysgen[:,3]*sysy2,atol=1.e-7) )
pause() 

println("\nSolution of a H∞ model matching problem  min||X(s)*G(s)-F(s)||")
println("Setting up an example taken from Francis' book (1987)")
println("s = Polynomial([0, 1],'s');")
println("W = (s+1)/(10s+1); # weighting function") 
println("G = dss([ -(s-1)/(s^2+s+1)*W; (s^2-2*s)/(s^2+s+1)*W], minimal=true);")
println("F = dss([ W; 0 ]);")

s = Polynomial([0, 1],'s'); 
W = (s+1)/(10s+1); # weighting function
G = dss([ -(s-1)/(s^2+s+1)*W; (s^2-2*s)/(s^2+s+1)*W], minimal=true);
F = dss([ W; 0 ]);
pause() 

println("\nWe employ the γ-iteration based solution proposed by Francis (1987)")

println("Xopt, info = grasol(G, F, mindeg = true, atol = 1.e-7);")
Xopt, info = grasol(G, F, mindeg = true, atol = 1.e-7); 
println("Optimum of approximation error: info.mindist = $(info.mindist)")
pause() 

println("This fully agrees with info.mindist: ghinfnorm(G*Xopt-F) ≈ info.mindist")
println(@test ghinfnorm(G*Xopt-F)[1] ≈ info.mindist)
pause() 

println("\nThe solution in Francis' book corresponds to a suboptimal solution for γ = 0.2729")
println("Xsub, info = grasol(G, F, 0.2729; mindeg = true, atol = 1.e-7);")
Xsub, info = grasol(G, F, 0.2729; mindeg = true, atol = 1.e-7);
println("The suboptimal approximation error: info.mindist = $(info.mindist)")
pause() 

println("This fully agrees with info.mindist: ghinfnorm(G*Xsub-F) ≈ info.mindist")
println(@test ghinfnorm(G*Xsub-F)[1] ≈ info.mindist)
pause() 



println("\nLet's illustrate the use of the bilinear transformation") 

println("s = rtf('s');")
println("G  = [s^2 s/(s+1); 0 1/s]")
println("sys = dss(G, minimal = true);")
s = rtf('s');                 # define the complex variable s
G  = [s^2 s/(s+1); 0 1/s]     # define the 2-by-2 improper G(s)
sys = dss(G, minimal = true); # build continuous-time descriptor system realization
pause() 

println("Pole-zero analysis shows that G(s) has poles and zeros in the origin and at infinity")
println("pol = gpole(sys);")
println("zer = gzero(sys);")
pol = gpole(sys);
zer = gzero(sys);
println("pol = ")
display(pol)
println("zer = ")
display(zer)
pause()

println("Define a bilinear transformation with g(s) = (s+0.01)/(1+0.01*s) to make") 
println("all poles and zeros stable and finite using") 
println("g = (s+0.01)/(1+0.01*s);")
g = (s+0.01)/(1+0.01*s);
println("\ng = \n")
print(g)
pause()

println("Compute the transformed system and resulting stable poles and zeros") 
println("syst = gbilin(sys,g, atol = 1.e-7,minimal = true)[1];")
syst = gbilin(sys,g, atol = 1.e-7,minimal = true)[1];
pol_new = gpole(syst)
zer_new = gzero(syst)
println("pol_new = ")
display(pol_new)
println("zer_new = ")
display(zer_new)
pause()

println("Compute the ν-gap distance between models using Vinnicombe's formula")
println("nugap = gnugap(sys,syst)[1]")
nugap = gnugap(sys,syst)[1]
println("The resulting ν-gap distance $nugap reflects well to the s-variable perturbation")

end # module
