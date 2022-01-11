module Varpro
using LinearAlgebra
using SparseArrays

include("VarproTypes.jl")

using NL2sol

export varpro, FitContext, NL2SOL, NL2SNO

"""
  # Description

    varpro solves a separable nonlinear least squares problem.
    
        varpro(ctx::FitContext)
        Returns: [alpha, c, wresid, wresid_norm, y_est, Regression]
    
        ctx is constructed by the FitContext() constructor.  See types.jl
        for more info
    
    Given a set of m observations y[1],...,y[m]
    this program computes a weighted least squares fit using the model
    
        eta(alpha,c,t) = 
            c_1 * phi_1 (alpha,t) + ...  + c_n * phi_n (alpha,t) 
        (possibly with an extra term  + phi_{n+1} (alpha,t) ).
    
    This program determines optimal values of the q nonlinear parameters
    alpha and the n linear parameters c, given observations y at m
    different values of the "time" t and given evaluation of phi and 
    (optionally) derivatives of phi.
    
    Varpro calls either NL2sol, NL2sno (fd Jacobian) which solves a 
    non-linear least squares problem.

    What distinguishes varpro from levenberg_marquardt is that, for efficiency and
    reliability, varpro causes the nonlinear solver NL2sol to only iterate on the
    nonlinear parameters.  Given the information in phi and dphi, this 
    requires an intricate but inexpensive computation of partial 
    derivatives, and this is handled by the varpro function formJacobian.
    
    nl2sol is in the NL2sol.jl package.
    
    The original Fortran implementation of the variable projection 
    algorithm (ref. 2) was modified in 1977 by John Bolstad 
    Computer Science Department, Serra House, Stanford University,
    using ideas of Linda Kaufman (ref. 5) to speed up the 
    computation of derivatives.  He also allowed weights on
    the observations, and computed the covariance matrix.
    
    The orginal MATLAB version of the Varpro program is documented in
    Dianne P. O'Leary and Bert W. Rust,
    "Variable Projection for Nonlinear Least Squares Problems",
    Computational Optimization and Applications (2012)
    doi 10.1007/s10589-012-9492-9.
    
    US National Inst. of Standards and Technology, 2010.
    
    Main reference:
    
        0.  Gene H. Golub and V. Pereyra, 'Separable nonlinear least   
            squares: the variable projection method and its applications,'
            Inverse Problems 19, R1-R26 (2003).
             
    See also these papers, cited in John Bolstad's Fortran code:
                                                                           
    1.  Gene H. Golub and V. Pereyra, 'The differentiation of      
        pseudo-inverses and nonlinear least squares problems whose 
        variables separate,' SIAM J. Numer. Anal. 10, 413-432      
        (1973).                                                    
    2.  ------, same title, Stanford C.S. Report 72-261, Feb. 1972.
    3.  Michael R. Osborne, 'Some aspects of non-linear least      
        squares calculations,' in Lootsma, Ed., 'Numerical Methods 
        for Non-Linear Optimization,' Academic Press, London, 1972.
    4.  Fred Krogh, 'Efficient implementation of a variable projection
        algorithm for nonlinear least squares problems,'           
        Comm. ACM 17:3, 167-169 (March, 1974).                    
    5.  Linda Kaufman, 'A variable projection method for solving  
        separable nonlinear least squares problems', B.I.T. 15,   
        49-57 (1975).                                             
    6.  C. Lawson and R. Hanson, Solving Least Squares Problems,
        Prentice-Hall, Englewood Cliffs, N. J., 1974.          

    These books discuss the statistical background:

    7.  David A. Belsley, Edwin Kuh, and Roy E. Welsch, Regression 
        Diagnostics, John Wiley & Sons, New York, 1980, Chap. 2.
    8.  G.A.F. Seber and C.J. Wild, Nonlinear Regression,
        John Wiley & Sons, New York, 1989, Sec. 2.1, 5.1, and 5.2.

    Dianne P. O'Leary, NIST and University of Maryland, February 2011.
    Bert W. Rust,      NIST                             February 2011.
    Comments updated 07-2012.

    MODIFIED: May 2016
    Ported to Julia, extended to handle complex residuals of complex
    parameters (both linear and non-linear), restructured to use context 
    struct, style changes and added ability to call nl2sol as the nonlinear
    solver

    Don MacMillen     
  
# Input

    On Input and stored in the FitContext struct ctx are:

    y    m x 1   vector containing the m observations (may be complex)
    w    m x 1   vector of weights used in the least squares
        fit.  We minimize the norm of the weighted residual
        vector r, where, for i=1:m,

        r[i] = w[i] * (y[i] - eta(alpha, c, t[i,:])).

        Therefore, w[i] should be set to 1 divided by
        the standard deviation in the measurement y[i].  
        If this number is unknown, set w[i] = 1.
        Only real weights are supported

    alpha q x 1  initial estimates of the parameters alpha.
        If alpha = [], Varpro assumes that the problem
        is linear and returns estimates of the c parameters.
        May be complex
    n            number of linear parameters c
    ada          a function handle, described below.
    gada         a function handle, described below.
    lb    q x 1  lower bounds on the parameters alpha. 
        (Optional)   (Omit this argument or use [] if there are
        no lower bounds.)  May be complex but currently unimplmented
    ub    q x 1  upper bounds on the parameters alpha. 
        (Optional)   (Omit this argument or use [] if there are
        no upper bounds.)  May be complex but currently unimplemented

# Output

    alpha  q      estimates of the nonlinear parameters.
    c      n      estimates of the linear parameters.
    wresid m      weighted residual vector, with i-th component
        w(i) * (y(i) - eta(alpha, c, t(i,:))).
    wresid_norm   norm of wresid.
    y_est  m      the model estimates = eta(alpha, c, t(i,:)))
    regression    a structure containing diagnostics about the model 

                **************************************************
                *                C a u t i o n:                  *
                *   The theory that makes statistical            *
                *   diagnostics useful is derived for            *
                *   linear regression, with no upper- or         *
                *   lower-bounds on variables.                   *
                *   The relevance of these quantities to our     *
                *   nonlinear model is determined by how well    *
                *   the linearized model (Taylor series model)   *
                *         eta(alpha_true, c_true)                *
                *            +  phi * (c  - c_true)              *
                *            + dphi * (alpha - alpha_true)       *
                *   fits the data in the neighborhood of the     *
                *   true values for alpha and c, where phi       *
                *   and dphi contain the partial derivatives     *
                *   of the model with respect to the c and       *
                *   alpha parameters, respectively, and are      *
                *   defined in ada.                              *
                **************************************************

    regression.results:  
        This structure includes information on the solution
        process, including the number of iterations, 
        termination criterion, and exitflag from lsqnonlin.

    regression.rank is the computed rank of the 
        matrix for the linear subproblem.  If this equals
        n, then the linear coefficients are well-determined.
        If it is less than n, then although the model might
        fit the data well, other linear coefficients might
        give just as good a fit.

    regression.sigma:        
        The estimate of the standard deviation is the
        weighted residual norm divided by the square root
        of the number of degrees of freedom.
        This is also called the "regression standard error"
        or the square-root of the weighted SSR (sum squared
        residual) divided by the square root of the
        number of degrees of freedom.

    regression.rms:
       The "residual mean square" is equal to sigma^2:
       RMS = wresid_norm^2 / (m-n+q)

    regression.coef_determ:
        The "coefficient of determination" for the fit,
        also called the square of the multiple correlation
        coefficient, is sometimes called R^2.
        It is computed as 1 - wresid_norm^2/CTSS,
        where the "corrected total sum of squares"
        CTSS is the norm-squared of W*(y-y_bar),
        and the entries of y_bar are all equal to
        (the sum of W_i^2 y_i) divided by (the sum of W_i^2).
        A value of .95, for example, indicates that 95 per 
        cent of the CTTS is accounted for by the fit.

    regression.covmx: (n+q) x (n+q)
        This is the estimated variance/covariance matrix for
        the parameters.  The linear parameters c are ordered
        first, followed by the nonlinear parameters alpha.
        This is empty if dphi is not computed by ada.

    regression.cormx: (n+q) x (n+q)
        This is the estimated correlation matrix for the 
        parameters.  The linear parameters c are ordered
        first, followed by the nonlinear parameters alpha.
        This is empty if dphi is not computed by ada.

    regression.std_param: (n+q) x 1
        This vector contains the estimate of the standard 
        deviation for each parameter.
        The k-th element is the square root of the k-th main 
        diagonal element of Regression.CovMx
        This is empty if dphi is not computed by ada.

    regression.t_ratio:   (n+q)
        The t-ratio for each parameter is equal to the
        parameter estimate divided by its standard deviation.
        (linear parameters c first, followed by alpha)
        This is empty if dphi is not computed by ada.

    regression.standardized_wresid:
        The k-th component of the "standardized weighted 
        residual" is the k-th component of the weighted 
        residual divided by its standard deviation.
        This is empty if dphi is not computed by ada.

    Specification of the function ada, which computes information
    related to phi:

    function  phi, dphi, ind = ada(alpha)

    This function computes phi and returns it.  Additionally, it
    computes more information that is needed in formJacobian and
    stores it in the context

    On Input: 

        alpha q     contains the current value of the alpha parameters.

        Note:  If more input arguments are needed, use the standard
               Matlab syntax to accomplish this.  For example, if
               the input arguments to ada are t, z, and alpha, then
               before calling varpro, initialize t and z, and in calling 
               varpro, replace "@ada" by "@(alpha)ada(t,z,alpha)".

     On Output:

        phi   m x n1   where phi(i,j) = phi_j(alpha,t_i).
                       (n1 = n if there is no extra term; 
                        n1 = n+1 if an extra term is used)
        dphi  m x p    where the columns contain partial derivative
                       information for phi and p is the number of 
                       columns in Ind 
                       (or dphi = [] if derivatives are not available).
        Ind   2 x p    Column k of dphi contains the partial
                       derivative of phi_j with respect to alpha_i, 
                       evaluated at the current value of alpha, 
                       where j = Ind(1,k) and i = Ind(2,k).
                       Columns of dphi that are always zero, independent
                       of alpha, need not be stored. 
        Example:  if  phi_1 is a function of alpha_2 and alpha_3, 
                  and phi_2 is a function of alpha_1 and alpha_2, then 
                  we can set
                          Ind = [ 1 1 2 2
                                  2 3 1 2 ]
                  In this case, the p=4 columns of dphi contain
                          d phi_1 / d alpha_2,
                          d phi_1 / d alpha_3,
                          d phi_2 / d alpha_1,
                          d phi_2 / d alpha_2,
                  evaluated at each t_i.
                  There are no restrictions on how the columns of
                  dphi are ordered, as long as Ind correctly specifies
                  the ordering.

        If derivatives dphi are not available, then set dphi = Ind = [].
      

"""
function varpro(ctx)
    y, W, q, m, n1, n =  ctx.y, ctx.w, ctx.q, ctx.m, ctx.n1, ctx.n

    T = eltype(ctx.alpha)
    nl2_msg = ""

    #
    # Solve the least squares problem using NL2sol or, if there
    # are no nonlinear parameters, using the SVD procedure in formJacobian.
    #
    regression = Regression(T, q, n, m)

    if q > 0  # The problem is nonlinear.
        
        if ctx.iscomplex
            alpha_real = [real(ctx.alpha); imag(ctx.alpha)]
            mreal = 2 * ctx.m
        else
            alpha_real = ctx.alpha
            mreal = ctx.m
        end

        f(a, r) = f_lsq(a, r, ctx)
        g(a, j) = g_lsq(a, j, ctx)
        iv, v = nl2_set_defaults(mreal, length(alpha_real))
        iv[MXFCAL] = ctx.mxfcal
        iv[MXITER] = ctx.mxiter
        iv[PRUNIT] = 0
        if ctx.opto == NL2SOL
            results = nl2sol(f, g, alpha_real, mreal, iv, v)
            ctx.verbose && println("\nNL2sol return code: ", return_code[iv[1]])
        else
            results = nl2sno(f, alpha_real, mreal, iv, v)
            ctx.verbose && println("\nNL2sno return code: ", return_code[iv[1]])
        end

        ctx.verbose && println(nl2_msg)
        ctx.verbose && @show(results)
        alpha_real .= results.minimum
        if ctx.iscomplex
            ctx.alpha .= complex.(alpha_real[1:ctx.q], alpha_real[ctx.q+1:end])  ## cwt barks here
        end
        wresid_norm2 = results.f_minimum
        f_lsq(alpha_real, ctx.wresid_real, ctx)
        g_lsq(alpha_real, ctx.jac_real,  ctx)
        r, Jacobian, phi, dphi, y_est = ctx.wresid, ctx.jac, ctx.phi, ctx.dphi, ctx.y_est
                                               
        wresid = r
        wresid_norm = sqrt(wresid_norm2)
        regression.results = results
        regression.rank = ctx.rank

    else       # The problem is linear.

        ctx.verbose && println("VARPRO problem is linear, since length(alpha) = 0")

        Jacobian = formJacobian(ctx)
        c = ctx.c
        wresid = ctx.wresid
        y_est = ctx.y_est
        regression.report.rank = ctx.rank
        wresid_norm = norm(wresid)
        wresid_norm2 = wresid_norm * wresid_norm
    end
        
    # Calculate sample variance,  the norm-squared of the residual
    #    divided by the number of degrees of freedom.
    sigma2 = wresid_norm2 / float(m - n - q)

    # Compute  Regression.sigma:        
    #     square-root of weighted residual norm squared divided 
    #     by number of degrees of freedom.
    regression.sigma = sqrt(sigma2)

    # Compute Regression.coef_determ:
    #     The coeficient of determination for the fit, also called the square
    #     of the multiple correlation coefficient, or R^2. It is computed as
    #     1 - wresid_norm^2/CTSS, where the corrected total sum of squares
    #     CTSS is the norm-squared of W*(y-y_bar), and the entries of y_bar 
    #     are all equal to (the sum of W_i y_i) divided by (the sum of W_i).

    w = diag(W)
    y_bar = sum(w.*y) / sum(w)
    CTTS = norm(W * (y .- y_bar)) ^ 2
    regression.coef_determ = 1 - wresid_norm^2 / CTTS

    # Compute  regression.RMS = sigma^2:
    #     the weighted residual norm divided by the number of degrees of 
    #     freedom. RMS = wresid_norm / sqrt(m-n+q)

    regression.rms = sigma2

    if !isempty(dphi) && !ctx.skip_stats
        # Calculate the covariance matrix CovMx, which is sigma^2 times the
        # inverse of H'*H, where  
        #              H = W*[phi,J] 
        # contains the partial derivatives of  wresid  with
        # respect to the parameters in alpha and c.
    
        if ctx.iscomplex
            dphi = dphi[:,1:q]
        end
        xx, pp = size(dphi)
        J = zeros(eltype(ctx.alpha), m, q)
        for kk = 1:pp
            j = ctx.ind[1, kk]
            i = ctx.ind[2, kk]
            if j > n
                J[:, i] = J[:, i] + dphi[:, kk]
            else
                J[:, i] = J[:, i] + ctx.c[j] * dphi[:, kk]
            end
        end
        # Uses compact pivoted QR.
        F = qr(W*[phi[:, 1:n] J], Val(true))
        Qj = F.Q
        Rj = F.R
        Pj = F.p
        T2 = Rj \ diagm(ones((size(Rj, 1))))
        covmx = sigma2 * T2 * T2'
        regression.covmx[Pj, Pj] = covmx  # Undo the pivoting permutation.
   
        # Compute  regression.CorMx:        
        #     estimated correlation matrix (n+q) x (n+q) for the
        #     parameters.  The linear parameters are ordered
        #     first, followed by the nonlinear parameters.
        d = 1 ./ sqrt.(diag(regression.covmx))
        D = spdiagm(0 => d)
        regression.cormx = D * regression.covmx * D

        # Compute  regression.std_param:
        #     The k-th element is the square root of the k-th main 
        #     diagonal element of CovMx. 
        regression.std_param = sqrt.(diag(regression.covmx))

        # Compute  regression.t_ratio:
        #     parameter estimates divided by their standard deviations.
        regression.t_ratio = [ctx.c; ctx.alpha] .* d

        # Compute  regression.standardized_wresid:
        #     The k-th component is the k-th component of the
        #     weighted residual, divided by its standard deviation.
        #     Let X = W*[phi, J], 
        #        h[k] = k-th main diagonal element of covariance
        #               matrix for wresid
        #             = k-th main diagonal element of X*inv(X'*X)*X' 
        #             = k-th main diagonal element of Qj*Qj'
        #     Then the standard deviation is estimated by sigma*sqrt(1-h[k])

        temp = zeros(eltype(ctx.alpha), m)
        for k = 1:m
            temp[k] = (Qj[k, :] * Qj[k, :]')[1]
        end
        regression.std_wresid = wresid ./(regression.sigma*sqrt.(1 .- temp))
     end

    if ctx.verbose
        println(" ")
        println("VARPRO Results:")
        println(" Linear Parameters: ", ctx.c)
        
        println(" Nonlinear Parameters: ", ctx.alpha)
        
        
        println(" Norm-squared of weighted residual  = ", wresid_norm2)
        println(" Norm-squared of data vector        = ", norm(w.*y)^2)
        println(" Norm         of weighted residual  = ", wresid_norm)
        println(" Norm         of data vector        = ", norm(w.*y))
        println(" Expected error of observations     = ", regression.sigma)
        println(" Coefficient of determination       = ", regression.coef_determ)
    end

    return ctx.alpha, ctx.c, wresid, wresid_norm, y_est, regression
end



function update_alpha!(alpha_trial, ctx)
    if ctx.iscomplex
        # We do it this way since alpha_trial can be a NL2Array and we haven't
        # implemented slicing for that type.
        for i = 1:ctx.q
            ctx.alpha_real[i] = alpha_trial[i]
            ctx.alpha_real[i+ctx.q] = alpha_trial[i+ctx.q]
        end
        ctx.alpha = complex.(ctx.alpha_real[1:ctx.q], ctx.alpha_real[ctx.q+1:end])
    else
        # again, cannot use a colon assignment here
        for i = 1:ctx.q
            ctx.alpha[i] = alpha_trial[i]
        end
    end
    return nothing
end


"""
# Description
    This function is used by a nonlinear least squares solver NL2sol
    to compute wresid, the current estimate of the weighted residual 
    (but it is wrapped in a closure first so it is only a function 
     of alpha_trial and wresid).

# Input
    alpha_trial  trial vector of non-linear parameters
    ctx: a varpro context
    function f_lsq(alpha_trial, ctx)

# Output                                                       
    wresid.  Also calculates current estimates of c, phi, y_est, 
        and myrank and stores in the context struct

# NB
    It uses the user-supplied function ada

    An added complication is that both alpha_trial and r must be
    real for the optimizer.  However both of these values may be
    complex.  In that case, we check ctx to see and then recombine
    into the complex values for ada and then blast back to real
    before returning
"""
function f_lsq(alpha_trial, r, ctx)

    # Convert back to complex if necessary
    update_alpha!(alpha_trial, ctx)
    W = ctx.w

    # We determine the optimal linear parameters c for
    # the given values of alpha, the resulting residual and the y estimate
    #
    # We use the singular value decomposition to solve the linear least
    # squares problem
    #
    #    min_{c} || W resid ||.
    #       resid =  y - phi * c.
    #
    # If W*phi has any singular value less than m * eps() its largest singular value, 
    # these singular values are set to zero.
    # The following values on stored on the FitContext:
    #      c        n x 1 the optimal linear parameters for this choice of alpha.
    #      wresid   m x 1 the weighted residual = W(y - phi * c)
    #      y_est    m x 1 the model estimates = phi * c)
    #      rank     1 x 1 the rank of the matrix W*phi.

    ctx.phi = ctx.ada(ctx.alpha, ctx)
    U, S, V = svd(W * ctx.phi[:, 1:ctx.n], full=true)
    ctx.U = U
    ctx.s = S
    ctx.V = V

    if ctx.n > 0
        s = S   
    else    #  n = 0
        
        ctx.c = ctx.iscomplex ? Base.Complex{Float64}[] : Float64[]
        ctx.y_est .= ctx.phi
        ctx.wresid .= W * (y - ctx.y_est)
        r .= ctx.wresid
        ctx.rank = 1
        return ctx.wresid
    end

    # Time to "regularize"
    tol = ctx.m * eps()
    ctx.rank = sum(s .> tol*s[1] ) # number of singular values > tol*norm(W*phi)
    s = s[1:ctx.rank]

    if ctx.rank < ctx.n && ctx.verbose
        println("Warning from VARPRO:")
        println("   The linear parameters are currently not well-determined.")
        println("   The rank of the matrix in the subproblem is $(ctx.rank)") 
        println("   which is less than the $(ctx.n) linear parameters.")
    end

    yuse = copy(ctx.y)
    if ctx.n < ctx.n1
        yuse  = ctx.y - ctx.phi[:, n1]  # extra function phi(:,n+1)
    end
    temp  = U[:, 1:ctx.rank]' * (W * yuse)
    ctx.c = V[:, 1:ctx.rank] * (temp ./ s)
    ctx.y_est = ctx.phi[:, 1:ctx.n] * ctx.c
    ctx.wresid = W * (yuse - ctx.y_est)

    # This is most certainly wrong now, but we don't use the case n < n1
    if ctx.n < ctx.n1
        ctx.y_est += ctx.phi[:, ctx.n1]
    end
    if ctx.iscomplex
        ctx.wresid_real[1:ctx.m] = real(ctx.wresid)
        ctx.wresid_real[ctx.m+1:end] = imag(ctx.wresid)
    else
        ctx.wresid_real .= ctx.wresid
    end

    # Set r[:] for NL2sol. This is unrolled because r might be an NL2Array
    for i in 1:length(ctx.wresid_real)
        r[i] = ctx.wresid_real[i]
    end

    return ctx.wresid_real
end


function g_lsq(alpha_trial, jnl2, ctx)
    # function g_lsq(alpha_trial, ctx)
    #                                                       
    # Returns: jacobian, but also calculates dphi and stores it in ctx
    #
    # This is an 'extra' function that does not appear in the MATLAB version
    # of Varpro.  It is needed by levenberg_marquardt solver to compute the
    # the Jacobian matrix for the nonlinear parameters
    #
    # It also computes dphi the partial derivatives of phi (if available).
    # It uses the user-supplied function gada and the Varpro function formJacobian.
    #
    # We depend on f_lsq to update ctx.alpha and ctx.alpha_real.  In effect,
    # we assume that we never ask for a gradient unless we have first asked for
    # for the residual.
    f_lsq(alpha_trial, ctx.wresid_real, ctx)

    ctx.gada(ctx.alpha, ctx)
    formJacobian(ctx)

    if ctx.iscomplex
        # This is true because of Cauchy-Riemann Equations
        ctx.jac_real = [real(ctx.jac)  -imag(ctx.jac);
                        imag(ctx.jac)   real(ctx.jac)]
    else
        ctx.jac_real = ctx.jac
    end

    # Now must set the memory that nl2sol uses.
    m, n = size(ctx.jac_real)
    for j = 1:n
        for i in 1:m
            jnl2[i, j] = ctx.jac_real[i, j]
        end
    end

    return ctx.jac_real
end


"""    
    function formJacobian(ctx)
    
    Returns: Jacobian
    
    This function computes the Jacobian and dphi
    It is used by the functions Varpro and f_lsq.
    
    Notation: there are m data observations
                        n1 basis functions (columns of phi)
                        n linear parameters c
                            (n = n1, or n = n1 - 1)
                        q nonlinear parameters alpha
                        p nonzero columns of partial derivatives of phi 
    
    Input: (on ctx)
        alpha  q x 1   the nonlinear parameters,
        phi    m x n1  the basis functions phi(alpha),
        dphi   m x p   the partial derivatives of phi
    
    The variables W, y, q, m, n1, and n are also used.
    
    Output: 
    
        Jacobian  m x p the Jacobian matrix, with J[i, k] = partial
                    derivative of W resid[i] with respect to alpha[k].
""" 
function formJacobian(ctx)
    if ctx.q == 0 || isempty(ctx.dphi)
        ctx.jac = []
        return
    end

    # Compute the Jacobian.
    # There are two pieces, which we call Jac1 and Jac2,
    # with Jacobian = - (Jac1 + Jac2).
    #
    # The formula for Jac1 is (P D(W*phi) pinv(W*phi) y,
    #             and Jac2 is ((W*phi)^+})^T (P D(W*phi))^T y.
    # where  P           is the projection onto the orthogonal complement
    #                       of the range of W*phi,
    #        D(W*phi)    is the m x n x q tensor of derivatives of W*phi,
    #        pinv(W*phi) is the pseudo-inverse of W*phi.
    #  (See Golub&Pereyra (1973) equation (5.4).  We use their notational  
    #   conventions for multiplications by tensors.)
    #
    # Golub & Pereyra (2003), p. R5 break these formulas down by columns:
    #     The j-th column of Jac1 is P D_j pinv(W*phi) y
    #                             =  P D_j c             
    # and the j-th column of Jac2 is (P D_j pinv(W*phi))^T y,
    #                             =  (pinv(W*phi))^T D_j^T P^T y
    #                             =  (pinv(W*phi))^T D_j^T wresid.
    # where D_j is the m x n matrix of partial derivatives of W*phi
    #     with respect to alpha(j).
    #
    # We begin the computation by precomputing 
    #       Wdphi, which contains the derivatives of W*phi, and 
    #       Wdphi_r, which contains Wdphi' * wresid.

    Wdphi = ctx.w * ctx.dphi
    Wdphi_r = Wdphi' * ctx.wresid
    T2 = zeros(eltype(ctx.alpha), ctx.n1, ctx.q)
    ctemp = ctx.c
    if ctx.n1 > ctx.n
        ctemp = [ctemp; 1]
    end

    #   Now we work column-by-column, for j=1:q.
    #
    #   We form Jac1 = D(W*phi) ctemp.
    #   After the loop, this matrix is multiplied by 
    #        P = U[:, ctx.rank+1:m] * (U[:, ctx.rank+1:m]' * Jac1)
    #   to complete the computation.
    # 
    #   We also form  T2 = (D_j(W*phi))^T wresid  by unpacking
    #   the information in Wdphi_r, using ind.
    #   After the loop, T2 is multiplied by the pseudoinverse
    #       (pinv(W*phi))^T = U[:,1:ctx.rank] * diagm(1./s) * (V[:, 1:ctx.rank]'
    #   to complete the computation of Jac2.
    #   Note: if n1 > n, last row of T2 is not needed.   
    for j = 1:ctx.q                            # for each nonlinear parameter alpha[j]
        range = findall(ctx.ind[2, :] .== j)   # columns of Wdphi relevant to alpha[j]
        indrows = vec(ctx.ind[1, range])       # relevant rows of ctemp, need as vector
        ctx.jac1[:, j] = Wdphi[:, range] * ctemp[indrows]
        T2[indrows, j] = Wdphi_r[range]
    end

    ctx.jac1 .= ctx.U[:, ctx.rank+1:ctx.m] * (ctx.U[:, ctx.rank+1:ctx.m]' * ctx.jac1)
    T2 = diagm(0 => 1 ./ ctx.s[1:ctx.rank]) * (ctx.V[:, 1:ctx.rank]' * T2[1:ctx.n, :])
    ctx.jac2 .= ctx.U[:, 1:ctx.rank] * T2

    ctx.jac .= -(ctx.jac1 + ctx.jac2)

    if ctx.debug
        if ctx.neglect
            ctx.verbose && println("VARPRO norm(neglected Jacobian)/norm(Jacobian) = ",
                                   norm(ctx.jac2,"fro")/norm(ctx.jac,"fro"))
            ctx.verbose && println("neglecting Jac2")
            ctx.jac .= -ctx.jac1
        end
    end
    return ctx.jac
end

end # module Varpro
