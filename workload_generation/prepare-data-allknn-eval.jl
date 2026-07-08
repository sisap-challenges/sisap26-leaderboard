using JLD2, Random

rng = Xoshiro(5050)
m = 10_000
n_ = 6_000_000
mkpath("eval")
ofile = "eval/benchmark-eval-wikipedia-bge-m3.h5"
#n_ = 200_000
#ofile = "benchmark-dev-wikipedia-bge-m3-small.h5"

train, itest = jldopen("./emb-wiki-en-bge-m3.h5") do f
    X = f["emb"]
    n = size(X, 2)
    P = shuffle!(rng, collect(1:n))
    X[:, P[1:n_]], X[:, P[n_+1:n_+m]]
end

X = nothing

otest = jldopen("./emb-wiki-es-bge-m3.h5") do f
    X = f["emb"]
    n = size(X, 2)
    P = shuffle!(rng, collect(1:n))
    X[:, P[1:m]]
end

jldopen(ofile, "w") do f
    f["train"] = train
    f["itest/queries"] = itest
    f["otest/queries"] = otest
end
