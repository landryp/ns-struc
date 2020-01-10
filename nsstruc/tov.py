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
DEFAULT_MAX_NUM_R = 2000
DEFAULT_MAX_DR = 1e5 ### cm
DEFAULT_PRESSUREC2_TOL = 10 ### g/cm^3

DEFAULT_PROPS = ['R', 'M', 'Lambda']

#------------------------

DEFAULT_MIN_RHOC = 0.8
DEFAULT_MAX_RHOC = 12.0
DEFAULT_RHOC_RANGE = [DEFAULT_MIN_RHOC, DEFAULT_MAX_RHOC]

DEFAULT_RTOL = 0.1

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

def tov(
        efe,
        y0,
        r0,
        props=DEFAULT_PROPS,
        max_num_r=DEFAULT_MAX_NUM_R,
        max_dr=DEFAULT_MAX_DR,
        pressurec2_tol=DEFAULT_PRESSUREC2_TOL,
        verbose=False,
        warn=True,
    ):
    '''integrate the structure equations for these initial conditions'''

    ### set up integrator
    res = ode(efe)
    res.set_initial_value(y0, r0)

    ### initialize the integration loop
    i = 0
    p_ind = props.index('R') ### we can rely on this being present because it has to be for termination condition to make sense
    y_old = res.y[:]
    r_old = 0.
    while res.successful():
        if (res.y[p_ind] < pressurec2_tol): ### main termination condition
            break

        elif i < max_num_r: # stop integration when pressure vanishes (to within tolerance tol)
            if y_old[p_ind] != res.y[p_ind]:
                guess_dr = 1.5 * (res.t - r_old) / (y_old[p_ind]/res.y[p_ind] - 1) ### P * dR/dP ~ P * (R - R0) / (P - P0) ~ (R - R0) / (P0/P - 1)
                                                                  ### factor of 1.5 is so that we don't get stuck in Zeno's paradox
                dr = min(max_dr, max(r0, guess_dr))

            else:
                dr = max_dr

            y_old = res.y[:] ### update these for better estimates of slope moving forward
            r_old = res.t

            res.integrate(res.t + dr) ### actually integrate
            i += 1

        else:
            if warn:
                print('WARNING: integration did not find surface after %d iterations'%max_num_r)
            break

    else:
        if warn:
            print('WARNING: integration was not successful!')

    if verbose:
        print('integration took %d steps'%i)

    # CALCULATE NS PROPERTIES AT SURFACE

    # extrapolate to find the surface
    frac = res.y[p_ind] / (res.y[p_ind] - y_old[p_ind])
    R = (res.t-r_old)*frac + res.t
    y = (res.y-y_old)*frac + res.y

    return [VALS2MACROS[prop](R, y, props) for prop in props]

def foliate(
        efe,
        rho2p,
        p2rho,
        p2eps,
        p2cs2i,
        r0,
        props=DEFAULT_PROPS,
        m_ind=None,
        r_ind=None,
        max_num_r=DEFAULT_MAX_NUM_R,
        max_dr=DEFAULT_MAX_DR,
        pressurec2_tol=DEFAULT_PRESSUREC2_TOL,
        min_rhoc=DEFAULT_MIN_RHOC,
        max_rhoc=DEFAULT_MAX_RHOC,
        rtol=DEFAULT_RTOL,
        verbose=False,
        warn=True,
        min_props=None, # allow us to recurse without repeating work
        max_props=None,
    ):
    '''perform a bisection search until the macroscopic quantities are sampled densely enough'''

    # compute the properties at the lowest rhoc
    properties = []
    if min_props is None:
        pc = rho2p(min_rhoc)
        min_props = tov(
            efe,
            initconds(pc, p2eps(pc), p2cs2i(pc), min_rhoc, r0, props),
            r0,
            props=props,
            max_num_r=max_num_r,
            max_dr=max_dr,
            pressurec2_tol=pressurec2_tol,
            verbose=verbose,
        )

    # compute the properties at the highest rhoc
    if max_props is None:
        pc = rho2p(max_rhoc)
        max_props = tov(
            efe,
            initconds(pc, p2eps(pc), p2cs2i(pc), max_rhoc, r0, props),
            r0,
            props=props,
            max_num_r=max_num_r,
            max_dr=max_dr,
            pressurec2_tol=pressurec2_tol,
            verbose=verbose,
        )

    ### compute the mid point so we can estimate interpolator error
    mid_rhoc = 0.5*(min_rhoc + max_rhoc)
    # integrate properties at the bisection point
    pc = rho2p(mid_rhoc)
    mid_props = tov(
        efe,
        initconds(pc, p2eps(pc), p2cs2i(pc), mid_rhoc, r0, props),
        r0,
        props=props,
        max_num_r=max_num_r,
        max_dr=max_dr,
        pressurec2_tol=pressurec2_tol,
        verbose=verbose,
    )

    ### check to see whether we need to bisect
    ### we're doing a bisection search, so the linear interp between min_rhoc and max_rhoc is easy
    ### note, we check for interpolator error on all macroscopic properties
    amid_props = np.array(mid_props)
    bisect = np.any(np.abs(0.5*(np.array(max_props)+np.array(min_props)) -  amid_props) > rtol * amid_props)

    ### condition on whether we are accurate enough
    if bisect: ### we need to divide and recurse
        ### set up arguments for recursive calls
        args = (efe, rho2p, p2rho, p2eps, p2cs2i, r0)
        kwargs = {
            'props':props,
            'm_ind':m_ind,
            'r_ind':r_ind,
            'max_num_r':max_num_r,
            'max_dr':max_dr,
            'pressurec2_tol':pressurec2_tol,
            'rtol':rtol,
            'verbose':verbose,
            'warn':warn,
        }

        # low rhoc recursive call
        kwargs.update({
            'min_rhoc':min_rhoc,
            'max_rhoc':mid_rhoc,
            'min_props':min_props,
            'max_props':mid_props,
        })
        left = foliate(*args, **kwargs)

        # high rhoc recursive call
        kwargs.update({
            'min_rhoc':mid_rhoc,
            'max_rhoc':max_rhoc,
            'min_props':mid_props,
            'max_props':max_props,
        })
        right = foliate(*args, **kwargs)

        # return the result
        return left + right[1:] ### don't return the repeated mid-point

    else: ### we have converged within this range of rhoc
        return [[min_rhoc]+min_props, [mid_rhoc]+mid_props, [max_rhoc]+max_props]
