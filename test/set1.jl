# Generate synthetic data to fit to a sum of exponentials
function sumexp(a, b, t)
    m = length(t)
    n = length(a)
    length(b) == n || error("length of a and b must match")
    y = zeros(eltype(b), m)
    for i = 1:m
        for j = 1:n
            y[i] += a[j] * exp(-b[j]*t[i])
        end
    end
    y
end

function f_exp(alpha, ctx)
    for i in 1:ctx.m
        for j = 1:ctx.n
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

function f_exmpl(alpha, ctx)
    # This is an example from O'Leary and Rust
    for i in 1:ctx.m
        ctx.phi[i, 1] = exp(-alpha[2] * ctx.t[i]) * cos(alpha[3] * ctx.t[i])
        ctx.phi[i, 2] = exp(-alpha[1] * ctx.t[i]) * cos(alpha[2] * ctx.t[i])
    end
    return ctx.phi
end

function g_exmpl(alpha, ctx)
    for i in 1:ctx.m
        ctx.dphi[i, 1] = -ctx.t[i] * ctx.phi[i, 1]
        ctx.dphi[i, 2] = -ctx.t[i] * exp(-alpha[2] * ctx.t[i]) * sin(alpha[3] * ctx.t[i])
        ctx.dphi[i, 3] = -ctx.t[i] * ctx.phi[i, 2]
        ctx.dphi[i, 4] = -ctx.t[i] * exp(-alpha[1] * ctx.t[i]) * sin(alpha[2] * ctx.t[i])
    end
    return ctx.dphi
end

function rexp()
    # NOTE:  When there are too few data points, or the range of x 
    #        values does not sample enough to be able to detect the different
    #        exponentials, first Varpro will complain that "The linear parameters
    #        are currently not well determined" and if severe enough, the svd 
    #        will fail in f_lsq, ie the linear problem is ill-conditioned (less
    #        than full rank).
    #
    #        This is a well known issue for sums of exponentials.  Note that
    #        if we use LM instead of NL2SOL, it will fail earlier in LM (inside
    #        of some lapack code for the same reasons as above)

    t = collect(-1:.1:20)
    a = [1., 2., 3.]
    b = [4., 5., 6.]
    y = sumexp(a, b, t)
    ind = [1 2 3; 1 2 3]
    b_init = [10.2, 17.3, 8.4]
    w = ones(length(y))
    ctx = FitContext(y, t, w, b_init, 3, ind, f_exp, g_exp)
end


# Here we fit complex data
function cexp()
    t = collect(-1:.1:20)
    a = [1., 2., 3.]
    b = [.5im, 1.1im, 2.0im]
    y = sumexp(a, b, t)
    ind = [1 2 3; 1 2 3]
    b_init = [3.2im, 4.3im, 2.4im]
    #b_init = [4.5, 5., 6.]
    w = ones(length(y))
    ctx = FitContext(y, t, w, b_init, 3, ind, f_exp, g_exp)
end

# Problem from O'Leary and Rust:
#
# The data y(t) were generated using the parameters 
#         alphatrue = [1.0; 2.5; 4.0], ctrue = [6; 1].  
# Noise was added to the data, so these parameters do not provide
# the best fit.  The computed solution is:
#    Linear Parameters:
#      5.8416357e+00    1.1436854e+00 
#    Nonlinear Parameters:
#      1.0132255e+00    2.4968675e+00    4.0625148e+00 
# Note that there are many sets of parameters that fit the data well.
function example()
    t = [0; .1; .22; .31; .46; .50; .63; .78; .85; .97]
    y = [ 6.9842;  5.1851;  2.8907;  1.4199; -0.2473; 
               -0.5243; -1.0156; -1.0260; -0.9165; -0.6805]

    # The weights for the least squares fit are stored in w.
    w = [ 1.0; 1.0; 1.0; 0.5; 0.5; 1.0; 0.5; 1.0; 0.5; 0.5]
    b_init = [0.5; 2; 3]  # initial guess
    ind = [1 1 2 2; 2 3 1 2]
    ctx = FitContext(y, t, w, b_init, 2, ind, f_exmpl, g_exmpl)
end

# This is a very simple double complex exponential with no noise.
function double_exponential()
    t = collect(0:.05:10)
    a = [1., 2.]
    b = [1 - 1im, 0.8 - 2im]
    y = sumexp(a, b, t)
    ind = [1 2; 1 2]
    b_init = [1 - 1.2im, 1 - 2.3im] 
    w = ones(length(y))
    ctx = FitContext(y, t, w, b_init, 2, ind, f_exp, g_exp)
end

# Here we fit complex data
function ctoo()
    n = 1000
    t = collect(range(0.0; stop=4pi, length=n))
    a = [1., 2., 2.]
    b = [0.1 + 10im, 0.2 + 20im, 0.3 + 30im]
    y = sumexp(a, b, t)
    ind = [1 2 3; 1 2 3]
    b_init = [3.2im, 4.3im, 2.4im]
    w = ones(length(y))
    ctx = FitContext(y, t, w, b_init, 3, ind, f_exp, g_exp)
end

function h1_ringdown()
    n = 4
    h1 = readdlm("h1_whitened.txt")
    t = h1[:, 1]
    y = complex.(h1[:, 2])  # must be complex to match x_init
    w = ones(length(t))
    ind = [collect(1:n)'; collect(1:n)']
    rng = MersenneTwister(137)
    x_init = complex.(0.1*rand(rng, n), 2.0*rand(rng, n))  
    ctx = FitContext(y, t, w, x_init, n, ind, f_exp, g_exp)
end

# Expected results.  First member of tuple are the linear
# parameters.  The second contains the non-linear parameters.
correct = Dict{Function, Tuple}(rexp => ([1., 2., 3.], [4., 5., 6.]),
            cexp => ([1., 2., 3.], [.5im, 1.1im, 2.0im]),
            example => ([5.8416357, 1.1436854], [1.0132255, 2.4968675, 4.0625148]),
            double_exponential => ([1.0, 2.0], [1.0 - 1im, 0.8 - 2im]),
            ctoo => ([1.0, 2.0, 2.0], [0.1 + 10im, 0.2 + 20im, 0.3 + 30im]),
            h1_ringdown => ([1.29881 + 0.70304im, -2.6128 - 2.66289im,
                            -2.6128 + 2.66289im,  1.29881 - 0.70304im],
                            [112.083 - 1513.604im, 297.804 - 995.322im,
                             297.804 + 995.322im,   112.083 + 1513.604im]))

# a bogus lessthan for complex numbers so we get (hopefully) consistent ordering
const clt = (x, y) -> isapprox(real(x), real(y), atol=1e-4) ? imag(x) < imag(y) : real(x) < real(y)

function runone(mkctx; sno=false, verbose=false, atol=0.0)
    ctx = mkctx()
    verbose && println("\n####### Varpro on problem ", string(mkctx), " ########")
    ctx.verbose = verbose
    sno && (ctx.opto = NL2SNO)
    (alpha, c, wresid, resid_norm, y_est, regression) = try 
        varpro(ctx)
    catch exc
        println("Exception: ", exc)
        println("Failed: exception raised in $(string(mkctx))")
        return false
    end
    if !isapprox(sort(alpha, lt=clt), sort(correct[mkctx][2], lt=clt), atol=atol)
        println("Failed: Non-linear parameters out of range on problem $(string(mkctx))")
        println("        got: ", sort(alpha, lt=clt))
        println("   expected: ", sort(correct[mkctx][2], lt=clt))
        return false
    end
    if !isapprox(sort(c, lt=clt), sort(correct[mkctx][1], lt=clt), atol=atol)
        println("Failed: Linear parameters out of range on problem $(string(mkctx))")
        println("        got: ", sort(c, lt=clt))
        println("   expected: ", sort(correct[mkctx][1], lt=clt))
        return false
    end
    return true
end
