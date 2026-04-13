using Varpro
using PyPlot
using DelimitedFiles

function f_exp(alpha, ctx)
    for j = 1:ctx.n
        for i in 1:ctx.m
            ctx.phi[i, j] = exp(-alpha[j] * ctx.t[i])
        end
    end
    ctx.phi
end

function g_exp(alpha, ctx)
    for i in 1:ctx.m
        ctx.dphi[i, :] = -ctx.t[i] * ctx.phi[i, :]
    end
    ctx.dphi
end


n = 6
h1 = readdlm("test/h1_whitened.txt")
t = h1[:, 1]
y = complex.(h1[:, 2])  # must be complex to match x_init
w = ones(length(t))
ind = [collect(1:n)'; collect(1:n)']
x_init = complex.(0.1*rand(n), 2.0*rand(n))

ctx = FitContext(y, t, w, x_init, n, ind, f_exp, g_exp)
(alpha, c, wresid, resid_norm, y_est, regression) = varpro(ctx)


