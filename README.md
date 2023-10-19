# Rational Function Approximation with Positive-Normalized Denominators
This repository contains the code for our paper 
[Rational Function Approximation with Positive-Normalized Denominators](https://arxiv.org/abs/2310.12053).
It has usable code for rational function approximation using existing methods
like the AAA and SK algorithms, as well as our new
<I>Rational Bernstein Denominator algorithm</I>. It also contains
reproducible code for the experiments in our paper.

## Rational Function Approximation
Rational function approximation takes the form
$$f(x)\approx \frac{P_N(x)}{Q_M(x)}$$
where $P_N(x)$ and $Q_M(x)$ are polynomials of degree $N$ and $M$
respectively. Such an approximation is well known to perform
better than a polynomial or spline approximation with corresponding
degrees of freedom. The key issue with rational function approximation
is that they can create poles in the approximation domain. This
ability may be a positive or a negative, depending on if $f(x)$ contains
poles or not.

We are interested in the case in which $f(x)$ is a smooth function,
in which case, one must avoid poles in the approximation domain.
To that end, we introduce the <I>Rational Bernstein Denominator</I> algorithm.
Our method cannot produce poles in the approximation domain, 
guaranteeing smooth approximations. Since we still use the 
rational form, our algorithm leverages rational functions' flexibility while maintaining the robustness of polynomial approximation.
Our experiments show that our algorithm can produce a smooth
approximations, beating spline-based methods.

## Differential Equations
Consider the following differential equation
$$y'' + p(x)y' + q(x)y = f(x),\quad\text{for}\quad x\in[0,1].$$
Numerical spectral solvers solve such a differential equation by 
approximating the non-constant coefficients, $p(x)$ and $q(x)$,
as a polynomial. Importantly, the space and time complexity
of numerical spectral solvers grow with respect to the degree
of polynomial approximation used. A tradeoff must be made - high
degree approximation can return accurate solutions but requires
long time to solve or a low-degree approximation that returns
low accuracy solutions but in a quick time.

Instead, one can approximate the non-constant coefficients as
$$p(x) = \frac{P_{N}(x)}{Q_M(x)}\quad\text{and}\quad q(x) = \frac{R_N(x)}{Q_M(x)},$$
where $P_N$ and $R_N$ are polynomials of degree $N$ and
$Q_M$ is a polynomial of degree $M$. Therefore, we can more accurately represent the differential equation using lower-degree polynomials in the numerator and denominator and solve for the alternate equation
$$Q_M(x)y'' + P_N(x)y' + R_N(x)y = Q_M(x)f(x),\quad\text{for}\quad x\in[0,1].$$
Since the non-constant coefficients are now low-degree polynomials, this speeds up the numerical spectral solver's runtime without sacrificing the accuracy of the solutions.

However, we require a way to guarantee $Q_M(x) \neq 0$ in the domain for the differential equation to return correct results. As such, the <I>Rational Bernstein Denominator</I> algorithm provides a crucial building block, as it guarantees no poles inside the interval $[0, 1]$. 