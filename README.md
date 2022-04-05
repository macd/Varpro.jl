# Varpro: Variable Projection for Nonlinear Least Squares Problems

Model fitting is often approached as an optimization problem where the
sum of the model errors are minimized by optimizing the model
parameters.  If some of the model parameters are non-linear, then a
non-linear optimization algorithm must be used.  This can be
inefficient if some of the parameters are, in fact, linear.

The Varpro algorithm recasts the problem so that only the nonlinear
parameters need to be considered by the nonlinear optimizer.  For more
details see the references below and the embedded docs in the source
code.

This Julia code is a translation and extension the of the Matlab code
that can be found [here](http://www.cs.umd.edu/~oleary/software/varpro.m).
The extensions involve handling complex inputs and a complex model (although
the optimization objective function remains real since the objective is
essentially the L2 norm of the residual (error) vector).

## Usage

The best way to learn how to use Varpro is to read reference [4].  The usage
in Julia differs somewhat from the MATLAB version.  With this version, we
first set up a FitContext by calling the constructor as in the following

    ctx = FitContext(y, t, w, x_init, n, ind, f_exp, g_exp)

All of these are required parameters.  The vector **y** is the data we
wish to fit sampled at the times **t** (or whatever is the independent
variable).  The vector **w** is a weight vector for selectively
weighting the data, all ones is usually a good first step. The vector
**x\_init** is the starting guess for the nonlinear parameters.  Note
that both **y** and **x\_init** can be either real or complex, but
they both must share the same type.  The integer **n** is the number
of basis functions, which is also the number of linear parameters.
The matrix **ind** specifies the structure of the dphi matrix (see
[4]).  The functions **f\_exp** and **g\_exp** calculate the phi and
dphi matrices.

The following is a complete example of fitting the H1 strain ringdown
values of the recently discovered gravity wave GW150914 [5].


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

    """ Fit n complex exponentials to the measured data """
    function exp_fit(n, y, t)
        w = ones(length(t))
        ind = [collect(1:n)'; collect(1:n)']
        x_init = complex.(0.1*rand(n), 2.0*rand(n))  
        ctx = FitContext(y, t, w, x_init, n, ind, f_exp, g_exp)
        (alpha, c, wresid, resid_norm, y_est, regression) = varpro(ctx)
    end

    function main()
        h1 = readdlm("h1_whitened.txt")
        t = h1[:, 1]
        y = complex.(h1[:, 2])  # must be complex to match x_init
        x, c, r, r_norm, y_est, reg = exp_fit(6, y, t)
        println("Norm of residual error: ", r_norm)
        plot(t, real(y), "o", label="measured H1 strain")
        plot(t, real(y_est), label="modeled H1 strain")
        xlabel("Time")
        ylabel("Strain")
        title("H1 Ringdown Model")
        legend(loc="upper right")
        savefig("modeled_GW150914_strain.png")
    end

    main()

The above code produces the following figure:

![alt-text][ringdown]

[ringdown]: modeled_GW150914_strain.png "Greetings Programs!"

## Nonlinear optimizer

Varpro depends upon having an underlying nonlinear optimization algorithm.
Here we default to use [NL2sol](https://github.com/macd/NL2sol.f) a venerable 
FORTRAN code from the early 1980s. That code is compiled to **NL2sol_jll**
which is then further wrapped by [NL2sol.jl](https://github.com/macd/NL2sol.jl)
which is used here. The code and theory behind it are discussed in [6] and
[7]

## Limitations

Only supported in Julia 1.4 and later.

If you are fitting a sum of exponentials, a common use case for
Varpro, note that it can be a tricky business. If the optimzer takes a
step too large in one parameter, the other parameters may be swamped
and the problem doesn't have full rank at the current value of x.
Varpro tries to take care of this with the SVD and then regularizing
by throwing away small singular values and reducing the dimensionality
of the problem.  This doesn't always work.

## References

[1] Golub, G.H., Pereyra, V.: "The differentiation of pseudoinverses and 
    nonlinear least squares problems whose variables separate". SIAM Journal 
    on Numerical Analysis 10, pp 413-432 (1973)

[2] Golub, G.H., Pereyra, V.: "Separable nonlinear least squares: The variable 
    projection method and its applications". Inverse Problems 19 (2), R1–R26 (2003)

[3] Pereyra, V., Scherer, eds:  "Exponential Data Fitting and its Applications"
    Bentham Books, ISBN: 978-1-60805-048-2 (2010)

[4] Dianne P. O'Leary, Bert W. Rust: "Variable projection for nonlinear least squares
    problems".  Computational Optimization and Applications April 2013, Volume 54, 
    Issue 3, pp 579-593  Available [here](http://www.cs.umd.edu/~oleary/software/varpro.pdf)

[5] B. P. Abbott el. al. "ASTROPHYSICAL IMPLICATIONS OF THE BINARY BLACK HOLE MERGER GW150914" 
    The Astrophysical Journal Letters, Volume 818, Number 2

[6] J.E. Dennis, D.M. Gay, R.E. Welsch, "An Adaptive Nonlinear
Least-Squares Algorithm", ACM Transactions on Mathematical Software
(TOMS), Volume 7 Issue 3, Sept. 1981, pp 348-368, ACM New York, NY, USA
[see here](http://dl.acm.org/citation.cfm?id=355965&CFID=660003329&CFTOKEN=25049918)

[7] J.E. Dennis, D.M. Gay, R.E. Welsch, "Algorithm 573: NL2SOL—An Adaptive
Nonlinear Least-Squares Algorithm", ACM Transactions on Mathematical
Software (TOMS), Volume 7 Issue 3, Sept. 1981, pp 369-383, ACM New
York, NY, USA [see here](http://dl.acm.org/citation.cfm?id=355966)
