import numpy as np
import dedalus.public as d3


def solve_single_coefficient_bessel(approximator, a, m, Lx=0, Ux=1, Nx=256, dtype=np.double,
                                    n_eigenvals=20):
    """ Solves the eigenvalue problem for Bessel's differential equation with a single
        non-constant coefficient
    
        Parameters
        ----------
        approximator : Approximator
            Assumed to have a `numerator` and `denominator` method and is fitted to exp(2ax)
        a : float
            Scaling parameter in the differential equation
        m : float
            Parameter in the differential equation
        Lx : float, default=0
            Lower bound of the domain
        Ux : float, default=1
            Upper bound of the domain
        Nx : int, default=256
            Number of grid points
        dtype : data-type, default=np.double
            Data type to solve the differential equation in
        n_eigenvals : int, default=20
            Number of eigenvalues to compute
        
        Returns
        -------
        Dedalus Solver
    """
    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(Lx, Ux))
    
    # Fields
    x = dist.local_grid(xbasis)
    
    c1 = dist.Field(bases=xbasis)
    c1['g'] = approximator.numerator(x)

    c2 = dist.Field(bases=xbasis)
    c2['g'] = approximator.denominator(x)
    
    u = dist.Field(name='u', bases=xbasis)
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    k = dist.Field(name='k')

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ux = dx(u) + lift(tau_1) # First-order reduction
    uxx = dx(ux) + lift(tau_2)
    
    # Problem
    problem = d3.EVP([u, tau_1, tau_2], eigenvalue=k, namespace=locals())
    problem.add_equation("(k * c1 - (m ** 2) * c2) * u + c2 * uxx / (a ** 2) = 0")
    problem.add_equation("u(x=Lx) = 0")
    problem.add_equation("u(x=Ux) = 0")

    solver = problem.build_solver(max_ncc_terms=None, ncc_cutoff=1e-12)
    solver.solve_sparse(solver.subproblems[0], N=n_eigenvals, target=0)
    
    return solver


def solve_multiple_coefficient_bessel(approximator, a, m, Lx=0, Ux=np.log(2), Nx=256, dtype=np.double,
                                      n_eigenvals=20):
    """ Solves the eigenvalue problem for Bessel's differential equation with multiple
        non-constant coefficients
    
        Parameters
        ----------
        approximator : Approximator
            Assumed to have a `numerator` and `denominator` method and is fitted to the following 
            three functions:
            [(1 - np.exp(-a * x)) ** 2 / (a ** 2), np.exp(-a * x) * (1 - np.exp(-a * x)) / a, 
             (np.exp(a * x) - 1) ** 2]
        a : float
            Scaling parameter in the differential equation
        m : float
            Parameter in the differential equation
        Lx : float, default=0
            Lower bound of the domain
        Ux : float, default=1
            Upper bound of the domain
        Nx : int, default=256
            Number of grid points
        dtype : data-type, default=np.double
            Data type to solve the differential equation in
        n_eigenvals : int, default=20
            Number of eigenvalues to compute
        
        Returns
        -------
        Dedalus Solver
    """
    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(Lx, Ux))
    
    # Fields
    x = dist.local_grid(xbasis)
    
    c1, c2, c3 = [dist.Field(bases=xbasis) for _ in range(3)]

    c1['g'], c2['g'], c3['g'] = approximator.numerator(x)

    den = dist.Field(bases=xbasis)
    den['g'] = approximator.denominator(x)
    
    u = dist.Field(name='u', bases=xbasis)
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    k = dist.Field(name='k')

    # Substitutions
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ux = dx(u) + lift(tau_1) # First-order reduction
    uxx = dx(ux) + lift(tau_2)

    # Problem
    problem = d3.EVP([u, tau_1, tau_2], eigenvalue=k, namespace=locals())
    problem.add_equation("c1 * uxx + c2 * ux + (k * c3 - (m ** 2) * den) * u = 0")
    problem.add_equation("u(x=Lx) = 0")
    problem.add_equation("u(x=Ux) = 0")

    solver = problem.build_solver(max_ncc_terms=None, ncc_cutoff=1e-12)
    solver.solve_sparse(solver.subproblems[0], N=n_eigenvals, target=0)
    
    return solver
