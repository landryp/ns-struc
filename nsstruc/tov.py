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
KNOWN_PROPS = ['R', 'M', 'Lambda', 'I', 'Mb']

#-------------------------------------------------

# DEFINE TOV AND PERTURBATION EQUATIONS

def hydro(r, y, eps, cs2i, rho, props): # hydrostatic equilibrium
    p = y[props.index('R')] # p in units of g/cm^3
    m = y[props.index('M')] # m in units of cm*c^2/G
    return -G*(eps(p)+p)*(m+4.*np.pi*r**3*p)/(c**2*r**2*(1.-2.*G*m/(c**2*r)))
               
def mass(r, y, eps, cs2i, Rho, props): # definition of the mass
    p = y[props.index('R')] # p in units of g/cm^3
    return 4.*np.pi*r**2*eps(p)
       
def baryonmass(r, y, eps, cs2i, rho, props): # definition of the baryonic mass
    p = y[props.index('R')] # p in units of g/cm^3
    m = y[props.index('M')] # m in units of cm*c^2/G

    f = 1.-2.*G*m/(c**2*r)
    return 4.*np.pi*r**2*rho(p)/f**0.5 
       
def equad(r, y, eps, cs2i, rho, props): # gravitoelectric quadrupole tidal perturbation
    p = y[props.index('R')] # p in units of g/cm^3
    m = y[props.index('M')] # m in units of cm*c^2/G
    eta = y[props.index('Lambda')] # dimensionless logarithmic derivative of metric perturbation

    f = 1.-2.*G*m/(c**2*r)
    A = (2./f)*(1.-3.*G*m/(c**2*r)-2.*G*np.pi*r**2*(eps(p)+3.*p)/c**2)
    B = (1./f)*(6.-4.*G*np.pi*r**2*(eps(p)+p)*(3.+cs2i(p))/c**2)

    return (-1./r)*(eta*(eta-1.)+A*eta-B) # from Landry+Poisson PRD 89 (2014)

def slowrot(r, y, eps, cs2i, rho, props): # slow rotation equation
    p = y[props.index('R')] # p in units of g/cm^3
    m = y[props.index('M')] # m in units of cm*c^2/G
    omega = y[props.index('I')] # log derivative of frame-dragging function

    f = 1.-2.*G*m/(c**2*r)
    P = 4.*np.pi*G*r**2*(eps(p)+p)/(c**2*f)

    return -(1./r)*(omega*(omega+3.)-P*(omega+4.))

EQSDICT = {
    'R': hydro,
    'M': mass,
    'Lambda': equad,
    'I': slowrot,
     'Mb': baryonmass,
}
def define_efe(eps, cs2i, rho, props):
    def efe(r, y):
        return [EQSDICT[prop](r, y, eps, cs2i, rho, props) for prop in props]
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

def tov(efe, y0, r0, props=DEFAULT_PROPS, num_r=DEFAULT_NUM_R, max_r=DEFAULT_MAX_R, pressurec2_tol=DEFAULT_PRESSUREC2_TOL):

    ### set up integrator
    res = ode(efe)
    res.set_initial_value(y0, r0)

    ### step size in radius
    max_dr = (max_r - r0) / num_r # fixed radial step size for data returned by integration

    i = 0
    p_ind = props.index('R')
#    p0 = y0[p_ind]
    while res.successful() and (res.y[p_ind] >= pressurec2_tol) and (i < num_r): # stop integration when pressure vanishes (to within tolerance tol)
#        guess_dr = 1.5 * res.t/(p0/res.y[p_ind] - 1) ### P * dR/dP ~ R / (P0/P - 1)
                                                      ### factor of 1.5 is so that we don't get stuck in Zeno's paradox
#        res.integrate(res.t + min(max_dr, max(r0, guess_dr)))

       	res.integrate(res.t + max_dr)
       	i += 1

    # CALCULATE NS PROPERTIES AT SURFACE
    return [VALS2MACROS[prop](res.t, res.y, props) for prop in props]
