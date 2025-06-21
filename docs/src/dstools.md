# System utilities

* **[`order`](@ref)**   Order of a system.
* **[`size`](@ref)**    Number of outputs and inputs of a descriptor system .
* **[`iszero`](@ref)**   Checking whether the transfer function matrix of a descriptor system is zero.
* **[`evalfr`](@ref)**   Gain of the transfer function matrix at a single frequency value.
* **[`dcgain`](@ref)**   DC gain of a system.
* **[`opnorm`](@ref)**   `L2`- and `L∞`-norms of a descriptor system.
* **[`rss`](@ref)**   Generation of randomized standard state-space systems.
* **[`rdss`](@ref)**   Generation of randomized descriptor state-space systems.
* **[`dsxvarsel`](@ref)**   Building a descriptor systems by selecting a set of state variables.
* **[`dssubset`](@ref)**   Assigning a subsystem to a given descriptor system.
* **[`dszeros`](@ref)**   Setting a subsystem to zero.
* **[`dssubsel`](@ref)**   Selecting a subsystem according to a given zero-nonzero pattern.
* **[`dsdiag`](@ref)**   Building a `k`-times diagonal concatenation of a descriptor system. 
* **[`dss2full`](@ref)**   Conversion of a descriptor state-space model to one with full matrices. 
* **[`dss2sparse`](@ref)**  Conversion of a descriptor state-space model to one with sparse matrices. 

```@docs
order
DescriptorSystems.size
DescriptorSystems.iszero
evalfr
dcgain
DescriptorSystems.opnorm
rss
rdss
dsxvarsel
dssubset
dszeros
dssubsel
dsdiag
dss2full
dss2sparse
```
