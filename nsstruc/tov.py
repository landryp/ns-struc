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

def hydro(r,y,*args): # hydrostatic equilibrium

       props = args[-1] # args expected as mu(p),1/cs2(p),rho(p),props

       p = y[props.index('R')] # p in units of g/cm^3
       m = y[props.index('M')] # m in units of cm*c^2/G
       mu = args[0] # mu in units of g/cm^3
       
       return -G*(mu(p)+p)*(m+4.*np.pi*r**3*p)/(c**2*r**2*(1.-2.*G*m/(c**2*r)))
               
def mass(r,y,*args): # definition of the mass
       
       props = args[-1]
       
       p = y[props.index('R')] # p in units of g/cm^3
       mu = args[0] # mu in units of g/cm^3    
               
       return 4.*np.pi*r**2*mu(p) 
       
def baryonmass(r,y,*args): # definition of the baryonic mass
       
       props = args[-1]
       
       p = y[props.index('R')] # p in units of g/cm^3
       m = y[props.index('M')] # m in units of cm*c^2/G
       rho = args[2] # rho in units of g/cm^3  
       
       f = 1.-2.*G*m/(c**2*r)
               
       return 4.*np.pi*r**2*rho(p)/f**0.5 
       
def equad(r,y,*args): # gravitoelectric quadrupole tidal perturbation
       
       props = args[-1]
       
       p = y[props.index('R')] # p in units of g/cm^3
       m = y[props.index('M')] # m in units of cm*c^2/G
       eta = y[props.index('Lambda')] # dimensionless logarithmic derivative of metric perturbation
       mu = args[0] # mu in units of g/cm^3
       cs2i = args[1] # sound speed squared in units of c^2
               
       f = 1.-2.*G*m/(c**2*r)
       A = (2./f)*(1.-3.*G*m/(c**2*r)-2.*G*np.pi*r**2*(mu(p)+3.*p)/c**2)
       B = (1./f)*(6.-4.*G*np.pi*r**2*(mu(p)+p)*(3.+cs2i(p))/c**2)
               
       return (-1./r)*(eta*(eta-1.)+A*eta-B) # from Landry+Poisson PRD 89 (2014)

def slowrot(r,y,*args): # slow rotation equation
       
       props = args[-1]
       
       p = y[props.index('R')] # p in units of g/cm^3
       m = y[props.index('M')] # m in units of cm*c^2/G
       omega = y[props.index('I')] # log derivative of frame-dragging function
       mu = args[0] # mu in units of g/cm^3
       
       f = 1.-2.*G*m/(c**2*r)  
       P = 4.*np.pi*G*r**2*(mu(p)+p)/(c**2*f)
               
       return -(1./r)*(omega*(omega+3.)-P*(omega+4.))  

def eqsdict(): # dictionary linking NS properties with corresponding equation of stellar structure

       return {'R': hydro,'M': mass,'Lambda': equad,'I': slowrot,'Mb': baryonmass}

#-------------------------------------------------

# INITIAL CONDITIONS

def initconds(pc, muc, cs2ic, rhoc, initial_r, props):
    '''initial conditions for integration of eqs of stellar structure'''

    Pc = pc - twopi * G_c2 * initial_r**2 * (pc + muc) * (3.*pc + muc) / 3.
    mc = fourpi * initial_r**3 * muc / 3.
    Lambdac = 2. + fourpi * G_c2 * initial_r**2 * (9.*pc + 13.*muc + 3.*(pc + muc)*cs2ic) / 21.
    omegac = 16.*pi * G_c2 * initial_r**2 * (pc + muc) / 5.
    mbc = fourpi * initial_r**3 * rhoc / 3.

    return {'R': Pc,'M': mc,'Lambda': Lambdac,'I': omegac, 'Mb': mc}

#-------------------------------------------------

# SURFACE VALUES

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

# INTERPOLATE CONTINUOUS FLUID VARIABLES FROM DISCRETE EOS DATA

def tov(mu, P, cs2i, Rho, rhoc, props=DEFAULT_PROPS, initial_r=DEFAULT_INITIAL_R, num_r=DEFAULT_NUM_R, max_r=DEFAULT_MAX_R, pressurec2_tol=DEFAULT_PRESSUREC2_TOL):
    stp = initial_r
    pts = num_r
    maxr = max_r
    tol = pressurec2_tol

    eqs = eqsdict() # associate NS properties with corresponding equation of stellar structure

    # PERFORM INTEGRATION OF EQUATIONS OF STELLAR STRUCTURE
    def efe(r,y): return [eqs[prop](r,y,mu,cs2i,Rho,props) for prop in props]

    pc = float(P(rhoc)) # central pressure from interpolated p(rho) function
    muc = mu(pc) # central energy density from interpolated mu(p) function
    cs2ic = cs2i(pc) # central sound speed from interpolated cs2i(p) function
    startvals = initconds(pc,muc,cs2ic,rhoc,stp,props) # load BCs at center of star for integration
    y0 = [startvals[prop] for prop in props]

    res = ode(efe)
    res.set_initial_value(y0,stp)
    dt = (maxr-stp)/pts # fixed radial step size for data returned by integration

    i=-1
    while res.successful() and res.y[props.index('R')] >= tol and i < pts-1: # stop integration when pressure vanishes (to within tolerance tol)
       	i = i+1
       	res.integrate(res.t+dt)

    # CALCULATE NS PROPERTIES AT SURFACE
    return [VALS2MACROS[prop](res.t, res.y, props) for prop in props]
