#!/usr/bin/python

import numpy as np
from scipy.special import hyp2f1
from .constants import *

# DEFINE TOV AND PERTURBATION EQUATIONS

# y = [p,m,r,eta,omega]
# args = [e(p),rho(p),1/cs2(p)]

def hydro(h,y,*args): # hydrostatic equilibrium

	idxs = args[-1]

	p = y[idxs['pR']] # pressure in g/cm^3
	e = args[0] # total energy density in g/cm^3

	return e(p) + p

def mass(h,y,*args): # definition of the mass

	idxs = args[-1]

	p = y[idxs['pR']] # pressure in g/cm^3
	m = y[idxs['M']] # mass in g
	r = y[idxs['R']] # radius in cm
	e = args[0] # total energy density in g/cm^3
	
	f = 1.-2.*G*m/(c**2*r)

	return -4.*np.pi*c**2*r**4*e(p)*f/(G*m+4.*np.pi*G*r**3*p)
		
def radius(h,y,*args): # definition of the radius

	idxs = args[-1]

	p = y[idxs['pR']] # pressure in g/cm^3
	m = y[idxs['M']] # mass in g
	r = y[idxs['R']] # radius in cm
	
	f = 1.-2.*G*m/(c**2*r)
	
	return -c**2*r**2*f/(G*m+4.*np.pi*G*r**3*p)
	
def baryonmass(h,y,*args): # definition of the baryonic mass
	
	idxs = args[-1]
	
	p = y[idxs['pR']] # pressure in g/cm^3
	m = y[idxs['M']] # mass in g
	r = y[idxs['R']] # radius in cm
	rho = args[1] # baryon density in g/cm^3	
	
	f = 1.-2.*G*m/(c**2*r)
		
	return -4.*np.pi*c**2*r**4*rho(p)*f**0.5/(G*m+4.*np.pi*G*r**3*p)
	
def equad(h,y,*args): # gravitoelectric quadrupole tidal perturbation
	
	idxs = args[-1]
	
	p = y[idxs['pR']] # pressure in g/cm^3
	m = y[idxs['M']] # mass in g
	r = y[idxs['R']] # radius in cm
	eta = y[idxs['Lambda']] # dimensionless logarithmic derivative of metric perturbation
	e = args[0] # total energy density in g/cm^3
	cs2i = args[2] # inverse sound speed squared in units of c^2
		
	f = 1.-2.*G*m/(c**2*r)
	A = 2.*r*(c**2/G-3.*m/r-2.*np.pi*r**2*(e(p)+3.*p))
	B = r*(6.*c**2/G-4.*np.pi*r**2*(e(p)+p)*(3.+cs2i(p)))
		
	return (eta*(eta-1.)*c**2*r*f/G+A*eta-B)/(m+4.*np.pi*r**3*p) # from Landry+Poisson PRD 89 (2014)

def slowrot(h,y,*args): # slow rotation equation
	
	idxs = args[-1]
	
	p = y[idxs['pR']] # pressure in g/cm^3
	m = y[idxs['M']] # mass in g
	r = y[idxs['R']] # radius in cm
	omega = y[idxs['I']] # log derivative of frame-dragging function
	e = args[0] # mu in units of g/cm^3
	
	f = 1.-2.*G*m/(c**2*r)	
	F = 4.*np.pi*r**3*(e(p)+p)/(-2*m+c**2*r/G)
		
	return (omega*(omega+3.)-F*(omega+4.))*c**2*r*f/(G*m+4.*np.pi*G*r**3*p)

def eqsdict(): # dictionary linking NS properties with corresponding equation of stellar structure

	return {'pR': hydro, 'M': mass, 'R': radius, 'Lambda': equad,'I': slowrot,'Mb': baryonmass}

# INITIAL CONDITIONS

def initconds(hc,pc,ec,rhoc,cs2ic,fracstp): # initial conditions for integration of eqs of stellar structure

	stp = hc*fracstp

	Pc = pc - stp*(ec+pc)
	rc = (3.*c**2*stp/(2.*np.pi*G*(ec+3.*pc)))**0.5
	mc = 4.*np.pi*rc**3*ec/3.
	mbc = 4.*np.pi*rc**3*rhoc/3.
	Lambdac = 2.+4.*np.pi*G*rc**2*(9.*pc+13.*ec+3.*(pc+ec)*cs2ic)/(21.*c**2)
	omegac = 16.*np.pi*G*rc**2*(pc+ec)/(5.*c**2)
	
	return {'pR': Pc, 'R': rc,'M': mc,'Lambda': Lambdac,'I': omegac, 'Mb': mbc}

# SURFACE VALUES

def calcobs(vals,idxs): # calculate NS properties at stellar surface in desired units, given output surficial values from integrator

	def psurf(vals): # error in surface pressure in g/cm^3
	
		return vals[idxs['pR']+1]

	def Rkm(vals): # R in km
	
		R = vals[idxs['R']+1]
	
		return R/1e5
		
	def MMsun(vals): # M in Msun
	
		M = vals[idxs['M']+1]
	
		return M/Msun
		
	def MbMsun(vals): # M in Msun
	
		Mb = vals[idxs['Mb']+1]
	
		return Mb/Msun
		
	def Lambda1(vals): # dimensionless tidal deformability
	
		etaR = vals[idxs['Lambda']+1] # log derivative of metric perturbation at surface
	
		C = G*vals[idxs['M']+1]/(c**2*vals[idxs['R']+1]) # compactness
		fR = 1.-2.*C
	
		F = hyp2f1(3.,5.,6.,2.*C) # a hypergeometric function
		
		def dFdz():

			z = 2.*C

			return (5./(2.*z**6.))*(z*(-60.+z*(150.+z*(-110.+3.*z*(5.+z))))/(z-1.)**3+60.*np.log(1.-z))
	
		RdFdr = -2.*C*dFdz() # log derivative of hypergeometric function
		
		k2el = 0.5*(etaR-2.-4.*C/fR)/(RdFdr-F*(etaR+3.-4.*C/fR)) # gravitoelectric quadrupole Love number
	
		return (2./3.)*(k2el/C**5)
		
	def MoI(vals):
	
		omegaR = vals[idxs['I']+1] # value of frame-dragging function at surface
	
		return 1e-45*(omegaR/(3.+omegaR))*c**2*vals[idxs['R']+1]**3/(2.*G) # MoI in 10^45 g cm^2

	return {'pR': psurf, 'R': Rkm,'M': MMsun,'Lambda': Lambda1,'I': MoI, 'Mb': MbMsun}	
