# As of this date (May 2016) nl2sno (finite difference calc of jacobian)
# is still encountering random memory corruption issues.  Use at you own
# risk...
@enum OptoAlgo  NL2SOL NL2SNO LEVENBERG


# What this macro does:
# Given an instance ctx of a type T with fields a, b, c, etc, it will
# make local variables a = ctx.a, b = ctx.b, etc by using the
# syntax    @vget(ctx, a, b, c)
# No error checking to see if a, b, and c are, in fact,
# fields of the type T.  Didn't use in Varpro
#
macro vget(ctx, args...)
    exps = []
    for i in 1:length(args)
        push!(exps, :($(esc(args[i])) = $ctx.($args[$i])))
    end
    return Expr(:block, exps...)
end

mutable struct Regression{T<:Number}
    sigma::T
    rms::T
    coef_determ::T
    covmx::Matrix{T}
    cormx::Matrix{T}
    std_param::Vector{T}
    t_ratio::Vector{T}
    std_wresid::Vector{T}
    results::Any
    rank::Int
end

function Regression(T, q, n, m)
    Regression(                          
        zero(T),              # sigma
        zero(T),              # rms
        zero(T),              # coef_determ
        zeros(T, q+n, q+n),   # covmx
        zeros(T, q+n, q+n),   # cormx
        zeros(T, m), 
        zeros(T, q+n), 
        zeros(T, m), 
        0,
        n
    )
end


# We intend T to be either Float64 or Complex128
mutable struct FitContext{T<:Number}
    # Required
    y::Vector{T}         # the m 'measurements'. Can be real or complex
    t::Vector{Float64}   # Independent variable (at m points) and is always real
    w::Matrix{Float64}   # the weight matrix (input initially as vector)
    alpha::Vector{T}     # Initial value of nonlinear variables. Can be real or complex
    n::Int               # number of linear parameters
    ind::Matrix{Int}     # integer matrix maps non zero entries of dphi
    ada::Function        # function to calculate phi
    gada::Function       # function to calculate dphi

    # Optional (only to change defaults)
    n1::Int              # Default to n
    debug::Bool          # turn on debugging
    neglect::Bool        # this turns on the Kaufman approximation to the jacobian
    opto::OptoAlgo       # choices are NL2SOL, NL2SNO (fd-jacobian), or LEVENBERG
    verbose::Bool        # More chatty if true
    mxfcal::Int          # Max number of function call for NL2sol
    mxiter::Int          # Max iterations for NL2sol

    # "Internal" values.  Some are for memory preallocation
    q::Int               # length(alpha)
    m::Int               # length(y)
    c::Vector{T}         # the linear variables
    y_est::Vector{T}     # current estimate of y
    rank::Int            # Rank of the phi matrix
    phi::Matrix{T}       # Current estimate of phi
    dphi::Matrix{T}      # current estimate of derivative of phi
    wresid::Vector{T}    # current weighted residual
    jac1::Matrix{T}      #
    jac2::Matrix{T}      #
    jac::Matrix{T}       #
    iscomplex::Bool      # are y and alpha complex?

    U::Matrix{T}         # SVD decomp of 
    s::Vector{T}
    V::Matrix{T}
    alpha_real::Vector{Float64}
    wresid_real::Vector{Float64}
    jac_real::Matrix{Float64}
end


function FitContext(y::Vector{T}, t, w, alpha::Vector{T}, n, ind, ada, gada) where {T}
    if length(size(y)) != 1 || length(size(t)) != 1 || length(size(w)) != 1 || 
       length(size(alpha)) != 1
        error("y, t, w, and alpha must all be vectors")
    end
    q = length(alpha)
    m = length(y)
    if length(t) != m || length(w) != m
        error("t (independent variable) and w (weight vector) must be same " *
              "size as y (sample vector)")
    end

    a, b = size(ind)
    if a != 2
        error("ind must by a 2 by mm array, where mm is the number of nonzero" *
              " columns of dphi")
    end

    if b > n*q
        error("The max size of the second dimension of ind is $n * $q")
    end
    n1 = n

    ctx = FitContext(
        # required
        y,                       # sample values to be fit
        t,                       # independent variable
        diagm(0 => w),                # weight matrix  (TODO: matlab code used spdiagm(w, 0, m, m))
        alpha,                   # current estimate on non-linear parameters
        n,                       # n
        ind,                     # ind::Matrix{Int}  constant integer matrix 
        ada,                     # phi and residual calculation
        gada,                    # dphi and jacobian calculation

        # Defaults 
        n1,                      # n1::Int
        false,                   # debug flag.
        false,                   # neglect jac2 in jacobian calculation
        NL2SOL,                  # default to nl2sol
        true,                    # default verbose on
        200,                     # default max func calls for NL2sol
        500,                     # default max iterations for NL2sol

        # "Internals"  
        q,                       # number of nonlinear parameters
        m,                       # number of samples
        zeros(T, n),             # c::Vector{T} current estimate of linear coefficients
        zeros(T, m),             # y_est::Vector{T}: current estimate of modeled y
        n,                       # rank::Int initialized to n

        zeros(T, m, n),          # phi::Matrix{T}    Current estimate of phi
        zeros(T, m, b),          # dphi::Matrix{T}   Current estimate of derivative of phi
        zeros(T, m),             # wresd::Vector{T}  Current weighted residual
        zeros(T, m, q),          # jac1::Matrix{T}
        zeros(T, m, q),          # jac2::Matrix{T}
        zeros(T, m, q),          # jac::Matrix{T} 
        T <: Complex,            # are y and t complex?

        zeros(T, m, n),          # U, a factor from the svd decomp
        zeros(T, n),             # s, a factor (as vector) from svd decomp
        zeros(T, n, n),          # V, a factor from the svd decomp

        # Need double size if y and alpha are complex
        zeros(Float64, T <: Complex ? 2*n : n),                         # alpha_real
        zeros(Float64, T <: Complex ? 2*m : m),                         # wresid_real
        zeros(Float64, T <: Complex ? 2*m : m, T <: Complex ? 2*size(ind,2) : 
              size(ind,2))  # jac_real
    )
end
