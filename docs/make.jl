using Documenter, DescriptorSystems
DocMeta.setdocmeta!(DescriptorSystems, :DocTestSetup, :(using DescriptorSystems); recursive=true)

makedocs(
  modules  = [DescriptorSystems],
  sitename = "DescriptorSystems.jl",
  authors  = "Andreas Varga",
  format   = Documenter.HTML(prettyurls = false),
  pages    = [
     "Home"   => "index.md",
     "Library" => [ 
        "dss.md",
        "operations.md",
        "connections.md",
        "order_reduction.md"
     ],
     "Utilities" => [
        "dstools.md"
     ],
     "Index" => "makeindex.md"
  ]
)

deploydocs(
  repo = "github.com/andreasvarga/DescriptorSystems.jl.git",
  target = "build",
)
