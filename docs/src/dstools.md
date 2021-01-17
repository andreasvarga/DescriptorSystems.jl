# Various descriptor system utilities

* **order**   Order (also the number of state variables) of a descriptor system.
* **size**   Number of outputs and inputs of a descriptor system .
* **iszero**   Checking whether the transfer function matrix of a descriptor system is zero.
* **evalfr**   Gain of the transfer function matrix of a descriptor system at a single frequency value.
* **dcgain**   DC gain of a descriptor system.
* **rss**   Generation of randomized standard state-space systems.
* **rdss**   Generation of randomized descriptor state-space systems.
* **gsvselect**   Building a descriptor systems by selecting a set of state variables.

```@docs
order
DescriptorSystems.size
DescriptorSystems.iszero
evalfr
dcgain
rss
rdss
gsvselect
```
