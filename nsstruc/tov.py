"a module that hoses logic for TOV integration"
__author__ = 'philippe.landry@ligo.org and Reed Essick (reed.essick@gmail.com)'

#-------------------------------------------------

import numpy as np

from scipy.interpolate import interp1d
from scipy.special import hyp2f1
from scipy.integrate import ode

### non-standard libraries
from .constants import *

#-------------------------------------------------

DEFAULT_INITIAL_R = 10 ### cm
DEFAULT_NUM_R = 2000
DEFAULT_MAX_R = 2e6 ### cm
DEFAULT_PRESSUREC2_TOL = 10 ### g/cm^3

DEFAULT_PROPS = ['R', 'M', 'Lambda']

#-------------------------------------------------

# DEFINE TOV AND PERTURBATION EQUATIONS

def hydro(r, p, m, eps): # hydrostatic equilibrium
    return -G_c2 * (eps + p) * (m + fourpi*p*r**3) / (r**2 * (1. - 2.*G_c2*m/r))
               
def mass(r, eps): # definition of the mass
    return fourpi * eps * r**2
       
def baryonmass(r, m, rho): # definition of the baryonic mass
    f = 1. - 2.*G_c2*m/r
    return fourpi * rho * r**2 / f**0.5 
       
def equad(r, p, m, eta, eps, cs2i): # gravitoelectric quadrupole tidal perturbation
    f = 1. - 2.*G_c2*m/r
    A = (2./f) * (1. - 3.*G_c2*m/r - 2.*G_c2*pi*r**2 * (eps + 3.*p))
    B = (1./f) * (6. - 4.*G_c2*pi*r**2 * (eps + p) * (3. + cs2i))
    return (-1./r) * (eta*(eta - 1.) + A*eta - B) # from Landry+Poisson PRD 89 (2014)

def slowrot(r, p, m, omega, eps): # slow rotation equation
    f = 1. - 2.*G_c2*m/r
    P = fourpi *G_c2*r**2*(eps + p) / f
    return -(1./r) * (omega*(omega + 3.) - P*(omega + 4.))

EQSDICT = {
    'R': hydro,
    'M': mass,
    'Lambda': equad,
    'I': slowrot,
    'Mb': baryonmass,
}
KNOWN_PROPS = list(EQSDICT.keys())
def define_efe(eps, cs2i, rho, props):
    """we jump through all these hoops to avoid having to repeatedly call props.index within the actual integration routine"""

    args = []
    p_ind = props.index('R') ### we can rely on this being present because we need it for termination condition...
    for prop in props:
        if prop == 'R':
            inds = [p_ind, props.index('M')]
            interps = (eps,)

        elif prop == 'M':
            inds = []
            interps = (eps,)

        elif prop == 'Mb':
            inds = [props.index('M')]
            interps = (rho,)

        elif prop == 'Lambda':
            inds = [p_ind, props.index('M'), props.index('Lambda')]
            interps = (eps, cs2i)

        elif prop == 'I':
            inds = [p_ind, props.index('M'), props.index('I')]
            interps = (eps,)

        else:
            raise ValueError('prop=%s not understood!'%prop)

        args.append((EQSDICT[prop], inds, interps))

    def efe(r, y):
        p = y[p_ind]
        return [foo(r, *list(y[inds])+[_(p) for _ in interps]) for foo, inds, interps in args]

    return efe

#-------------------------------------------------

# INITIAL CONDITIONS

def initconds(pc, epsc, cs2ic, rhoc, initial_r, props):
    '''initial conditions for integration of eqs of stellar structure'''

    Pc = pc - twopi * G_c2 * initial_r**2 * (pc + epsc) * (3.*pc + epsc) / 3.
    mc = fourpi * initial_r**3 * epsc / 3.
    Lambdac = 2. + fourpi * G_c2 * initial_r**2 * (9.*pc + 13.*epsc + 3.*(pc + epsc)*cs2ic) / 21.
    omegac = 16.*pi * G_c2 * initial_r**2 * (pc + epsc) / 5.
    mbc = fourpi * initial_r**3 * rhoc / 3.

    startvals = {'R': Pc,'M': mc,'Lambda': Lambdac,'I': omegac, 'Mb': mc}
    return [startvals[prop] for prop in props]

#-------------------------------------------------

# SURFACE VALUES to mactroscip observables

def vals2Rkm(R, vals, props): # R in km
    return R/1.e5

def vals2MMsun(R, vals, props): # M in Msun
    M = vals[props.index('M')]
    return M/Msun

def vals2MbMsun(R, vals, props): # M in Msun
    Mb = vals[props.index('Mb')]
    return Mb/Msun

def dFdz(z):
    return (5./(2.*z**6.)) * (z * (-60. + z*(150. + z*(-110. + 3.*z*(5. +z)))) / (z - 1.)**3 + 60.*np.log(1. - z))

def vals2Lambda(R, vals, props): # dimensionless tidal deformability
    etaR = vals[props.index('Lambda')] # log derivative of metric perturbation at surface
    C = G_c2 * vals[props.index('M')] / R # compactness
    fR = 1. - 2.*C
    F = hyp2f1(3., 5., 6., 2.*C) # a hypergeometric function

    RdFdr = -2. * C * dFdz(2.*C) # log derivative of hypergeometric function
    k2el = 0.5 * (etaR - 2. - 4.*C/fR) / (RdFdr - F*(etaR + 3. - 4.*C/fR)) # gravitoelectric quadrupole Love number
    return (2./3.) * (k2el / C**5)

def vals2MoI(R, vals, props):
    omegaR = vals[props.index('I')] # value of frame-dragging function at surface
    return 1e-45 * (omegaR / (3. + omegaR)) * R**3 / (2.*G_c2) # MoI in 10^45 g cm^2

VALS2MACROS = {
    'R': vals2Rkm,
    'M': vals2MMsun,
    'Lambda': vals2Lambda,
    'I': vals2MoI,
    'Mb': vals2MbMsun,
}

#-------------------------------------------------

# INTEGRATE TOV AND FRIENDS

def tov(efe, y0, r0, props=DEFAULT_PROPS, num_r=DEFAULT_NUM_R, max_r=DEFAULT_MAX_R, pressurec2_tol=DEFAULT_PRESSUREC2_TOL, verbose=True):

    ### set up integrator
    res = ode(efe)
    res.set_initial_value(y0, r0)

    ### step size in radius
    max_dr = (max_r - r0) / num_r # fixed radial step size for data returned by integration

    i = 0
    p_ind = props.index('R')
#    p0 = y0[p_ind]
    while res.successful():
        if (res.y[p_ind] < pressurec2_tol): ### main termination condition
            break

        elif i < num_r: # stop integration when pressure vanishes (to within tolerance tol)
#            guess_dr = 1.5 * res.t/(p0/res.y[p_ind] - 1) ### P * dR/dP ~ R / (P0/P - 1)
                                                          ### factor of 1.5 is so that we don't get stuck in Zeno's paradox
#            res.integrate(res.t + min(max_dr, max(r0, guess_dr)))

            res.integrate(res.t + max_dr)
            i += 1

        else:
            if verbose:
                print('WARNING: integration did not find surface after %d iterations'%num_r)
            break

    else:
        if verbose:
            print('WARNING: integration was not successful!')

    # CALCULATE NS PROPERTIES AT SURFACE
    return [VALS2MACROS[prop](res.t, res.y, props) for prop in props]
