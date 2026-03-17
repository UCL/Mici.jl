using Mici
using Documenter
using Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const SRC_DIR = joinpath(@__DIR__, "src")
const LITERATED_SUBDIR = "literated"
const OUTPUT_DIR = joinpath(SRC_DIR, LITERATED_SUBDIR)

const EXAMPLES_PAGES = [
    "Basic usage of package" => "simple_model_evaluation.jl",
]

pages = ["Home" => "index.md"]
for (title, example) in EXAMPLES_PAGES
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR; execute = true, flavor = Literate.DocumenterFlavor())
    literated_page = joinpath(LITERATED_SUBDIR, splitext(example)[1] * ".md")
    push!(pages, title => literated_page)
end

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
    pages,
)

deploydocs(; repo = "github.com/UCL/Mici.jl", devbranch = "main")
