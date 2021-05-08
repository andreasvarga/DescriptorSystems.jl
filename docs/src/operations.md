# Basic operations on system models

* **[`inv`](@ref)**  Inversion of a system.
* **[`ldiv`](@ref)**   Left division for two systems (also overloaded with **`\`**).
* **[`rdiv`](@ref)**   Right division for two systems (also overloaded with **`/`**).
* **[`gdual`](@ref)**   Building the dual of a descriptor system (also overloaded with **`transpose`**)
* **[`ctranspose`](@ref)**  Building the conjugate transpose of a system (also overloaded with **`adjoint`** and **`'`**).
* **[`adjoint`](@ref)**  Building the adjoint of a system.

```@docs
DescriptorSystems.inv
ldiv
rdiv
gdual
ctranspose
DescriptorSystems.adjoint
```
