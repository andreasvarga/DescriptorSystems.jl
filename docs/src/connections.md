# Interconnecting descriptor system models

* **append**  Building aggregate models by appending the inputs and outputs.
* **parallel**   Connecting models in parallel (also overloaded with **`+`**).
* **series**   Connecting models in series (also overloaded with **`*`**).
* **horzcat**   Horizontal concatenation of descriptor system models (also overloaded with **`[ * * ]`**).
* **vertcat**   Vertical concatenation of descriptor system models (also overloaded with **`[ *; * ]`**).

```@docs
append
parallel
series
horzcat
vertcat
```
