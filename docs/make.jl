using Mici
using Documenter

DocMeta.setdocmeta!(Mici, :DocTestSetup, :(using Mici); recursive = true)

makedocs(;
    modules = [Mici],
    authors = "Matt Graham, Ross Ah-Weng, Callum Lau, Anees Hussain, Jordan Simbananiye and contributors",
    sitename = "Mici.jl",
    format = Documenter.HTML(;
        canonical = "https://github-pages.ucl.ac.uk/Mici.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/UCL/Mici.jl", devbranch = "main")
