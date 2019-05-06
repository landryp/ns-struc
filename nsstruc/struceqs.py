#!/usr/bin/python

import numpy as np
from scipy.special import hyp2f1
from .constants import *

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

# INITIAL CONDITIONS

def initconds(pc,muc,cs2ic,rhoc,stp,props): # initial conditions for integration of eqs of stellar structure

	Pc = pc - 2.*np.pi*G*stp**2*(pc+muc)*(3.*pc+muc)/(3.*c**2)
	mc = 4.*np.pi*stp**3*muc/3.
	Lambdac = 2.+4.*np.pi*G*stp**2*(9.*pc+13.*muc+3.*(pc+muc)*cs2ic)/(21.*c**2)
	omegac = 0.+16.*np.pi*G*stp**2*(pc+muc)/(5.*c**2)
	mbc = 4.*np.pi*stp**3*rhoc/3.
	
	return {'R': Pc,'M': mc,'Lambda': Lambdac,'I': omegac, 'Mb': mc}

# SURFACE VALUES

def calcobs(vals,props): # calculate NS properties at stellar surface in desired units, given output surficial values from integrator

	def Rkm(vals): # R in km
	
		R = vals[0]
	
		return R/1e5
		
	def MMsun(vals): # M in Msun
	
		M = vals[props.index('M')+1]
	
		return M/Msun
		
	def MbMsun(vals): # M in Msun
	
		Mb = vals[props.index('Mb')+1]
	
		return Mb/Msun
		
	def Lambda1(vals): # dimensionless tidal deformability
	
		etaR = vals[props.index('Lambda')+1] # log derivative of metric perturbation at surface
	
		C = G*vals[props.index('M')+1]/(c**2*vals[0]) # compactness
		fR = 1.-2.*C
	
		F = hyp2f1(3.,5.,6.,2.*C) # a hypergeometric function
		
		def dFdz():

			z = 2.*C

			return (5./(2.*z**6.))*(z*(-60.+z*(150.+z*(-110.+3.*z*(5.+z))))/(z-1.)**3+60.*np.log(1.-z))
	
		RdFdr = -2.*C*dFdz() # log derivative of hypergeometric function
		
		k2el = 0.5*(etaR-2.-4.*C/fR)/(RdFdr-F*(etaR+3.-4.*C/fR)) # gravitoelectric quadrupole Love number
	
		return (2./3.)*(k2el/C**5)
		
	def MoI(vals):
	
		omegaR = vals[props.index('I')+1] # value of frame-dragging function at surface
	
		return 1e-45*(omegaR/(3.+omegaR))*c**2*vals[0]**3/(2.*G) # MoI in 10^45 g cm^2

	return {'R': Rkm,'M': MMsun,'Lambda': Lambda1,'I': MoI, 'Mb': MbMsun}	
