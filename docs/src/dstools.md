# System utilities

* **[`order`](@ref)**   Order of a system.
* **[`size`](@ref)**    Number of outputs and inputs of a descriptor system .
* **[`iszero`](@ref)**   Checking whether the transfer function matrix of a descriptor system is zero.
* **[`evalfr`](@ref)**   Gain of the transfer function matrix at a single frequency value.
* **[`freqresp`](@ref)**   Frequency response of a descriptor system.
* **[`timeresp`](@ref)**   Time response of a descriptor system.
* **[`dcgain`](@ref)**   DC gain of a system.
* **[`opnorm`](@ref)**   `L2`- and `L∞`-norms of a descriptor system.
* **[`rss`](@ref)**   Generation of randomized standard state-space systems.
* **[`rdss`](@ref)**   Generation of randomized descriptor state-space systems.
* **[`gsvselect`](@ref)**   Building a descriptor systems by selecting a set of state variables.

```@docs
order
DescriptorSystems.size
DescriptorSystems.iszero
evalfr
freqresp
timeresp
dcgain
DescriptorSystems.opnorm
rss
rdss
gsvselect
```
