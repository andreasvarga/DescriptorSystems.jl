# Interconnecting descriptor system models

* **[`append`](@ref)**  Building aggregate models by appending the inputs and outputs.
* **[`parallel`](@ref)**   Connecting models in parallel (also overloaded with **`+`**).
* **[`series`](@ref)**   Connecting models in series (also overloaded with **`*`**).
* **[`feedback`](@ref)**   Applying a static output feedback.
* **[`horzcat`](@ref)**   Horizontal concatenation of descriptor system models (also overloaded with **`[ * * ]`**).
* **[`vertcat`](@ref)**   Vertical concatenation of descriptor system models (also overloaded with **`[ *; * ]`**).

```@docs
append
parallel
series
feedback
horzcat
vertcat
```
