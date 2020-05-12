#  Well trying to use Varpro for Gaussian Mixture models is problematic.  We can
#  get away with that here because we use only use 3 single variate Gaussians
#  that are well separated and have comparable magnitudes.  Even then, I had to
#  run the random intial value version about a dozen times to find an initial starting
#  point that converges to the correct value

using Varpro
using Base.Iterators

phi(x, μ, σ) = exp(-((x - μ)/σ)^2/2)
g(x, μ, σ)   = phi(x, μ, σ) / (σ * sqrt(2π))
dμ(x, μ, σ) = (x - μ)   * phi(x, μ, σ) / (σ^3 * sqrt(2π))
dσ(x, μ, σ) = (x - μ)^2 * phi(x, μ, σ) / (σ^4 * sqrt(2π))

# We will use these params to form the synthetic gaussian mixture data.
# We have a mixture of 3 single variate gaussians and these are the
# parameters that we expect to recover using Varpro.
const mu   = [2.0, 3.0, 0.0]
const sig  = [0.5, 0.2, 1.0]
const cc   = [0.25, 0.25, 0.50]

# for creating synthetic data
function mixn(x)
    sum = 0.0
    for i in 1:length(cc)
        sum +=  cc[i] * g(x, mu[i], sig[i])
    end
    return sum
end

# The phi matrix.  Each row is for a different data point.  Each column
# is the contribution from a single gaussian
function f_gaussian(alpha, ctx)
    for j = 1:ctx.n
        for i in 1:ctx.m
           ctx.phi[i, j] = g(ctx.t[i], alpha[2j-1], alpha[2j])
        end
    end
    ctx.phi
end


function g_gaussian(alpha, ctx)
    for j = 1:ctx.n
        for i in 1:ctx.m
            ctx.dphi[i, 2j-1] = dμ(ctx.t[i], alpha[2j-1], alpha[2j])
            ctx.dphi[i, 2j]   = dσ(ctx.t[i], alpha[2j-1], alpha[2j])
        end
    end
    ctx.dphi
end


function testgauss(;verbose=true)
    n = 3  # length(c)  # the number of gaussian components

    # Our synthetic data
    t = [-10.0:0.01:10.0;]
    y = mixn.(t)
    m = length(t)
    
    # we might want to use different weights for different problems here
    w = ones(m)

    # this _sometimes_ converges to correct result
    #x_init = 2.0 * rand(2n)
    #@show x_init

    # carefully chosen starting point (actually one of the random ones that worked)
    x_init = [1.680, 0.744, 0.030, 0.312, 0.259, 1.427]
    
    # tell varpro how the jacobian is structured.
    ind = [collect(flatten((i,i) for i in 1:n))'; [1:2n;]']
    
    ctx = FitContext(y, t, w, x_init, n, ind, f_gaussian, g_gaussian)
    ctx.skip_stats = true
    ctx.verbose = verbose
    alpha, cfit, wresid, resid_norm, y_est, reg = varpro(ctx)
    return all(isapprox.(cc, sort(cfit))) && all(isapprox.(sort([mu;sig]), sort(alpha), atol=1e-13))
end
             
