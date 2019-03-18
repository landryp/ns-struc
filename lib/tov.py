#!/usr/bin/python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.special import hyp2f1
from geteos import geteos
from constants import G,c,Msun,rhonuc

# DEFINE TOV AND PERTURBATION EQUATIONS

#pos = dict(props[i]=i for i in range(len(props)))

def hydro(r,y,*args): # hydrostatic equilibrium

	props = args[-1]

	p = y[props.index('R')] # p in units of g/cm^3
	m = y[props.index('M')] # m in units of cm*c^2/G
	mu = args[0] # mu in units of g/cm^3
	
	return -G*(mu(p)+p)*(m+4.*np.pi*r**3*p)/(c**2*r**2*(1.-2.*G*m/(c**2*r)))
		
def mass(r,y,*args): # definition of the mass
	
	props = args[-1]
	
	p = y[props.index('R')] # p in units of g/cm^3
	m = y[props.index('M')] # m in units of cm*c^2/G
	mu = args[0] # mu in units of g/cm^3	
		
	return 4.*np.pi*r**2*mu(p) 
	
def equad(r,y,*args): # gravitoelectric quadrupole tidal perturbation
	
	props = args[-1]
	
	p = y[props.index('R')] # p in units of g/cm^3
	m = y[props.index('M')] # m in units of cm*c^2/G
	eta = y[props.index('Lambda')] # dimensionless logarithmic derivative of metric perturbation
	mu = args[0] # mu in units of g/cm^3
	cs2i = args[1] # mu in units of g/cm^3
		
	f = 1.-2.*G*m/(c**2*r)
	A = (2./f)*(1.-3.*G*m/(c**2*r)-2.*G*np.pi*r**2*(mu(p)+3.*p)/c**2)
	B = (1./f)*(6.-4.*G*np.pi*r**2*(mu(p)+p)*(3.+cs2i(p))/c**2)
		
	return (-1./r)*(eta*(eta-1.)+A*eta-B)

# INTERPOLATE FLUID VARIABLES FROM EOS DATA

def tov(eosloc,rhoc,props='R,M',stp=1e-4,pts=1e3,maxr=2e6,tol=1e-4):

	props = props.split(',')
	eqs = dict(R=hydro,M=mass,Lambda=equad)
	pts = int(pts)

	eos = geteos(eosloc)
	rhodat = eos[:,0] # rest-mass energy density in units of g/cm^3
	pdat = eos[:,1] # pressure in units of g/cm^3
	mudat = eos[:,2] # total energy density in units of g/cm^3

	mup = interp1d(pdat,mudat,kind='linear',bounds_error=False,fill_value=0.)
	
	def mu(p):
	
		return mup(p)
		
	prho = interp1d(rhodat,pdat,kind='linear',bounds_error=False,fill_value=0.)
		
	def P(rho):
	
		return prho(rho)
		
	cs2pi = interp1d(pdat,np.gradient(mudat)/np.gradient(pdat),kind='linear',bounds_error=False,fill_value=0.)
		
	def cs2i(p): # 1/sound speed squared
	
		return cs2pi(p)

# PERFORM INTEGRATION OF RELEVANT EQUATIONS
		
	def efe(r,y):
	
		return [eqs[prop](r,y,mu,cs2i,props) for prop in props]

	pc = float(P(rhoc))
	muc = mu(pc)
	startvals = dict(R=pc,M=4.*np.pi*stp**3*muc/3.,Lambda=2.)
	y0 = [startvals[prop] for prop in props]
	
	res = ode(efe)
	res.set_initial_value(y0,stp)
	tlist = np.logspace(np.log10(stp),np.log10(maxr),pts)
	dt = (maxr-stp)/pts
	
	sols = np.zeros((len(props)+1,pts))
	i=-1
	
	while res.successful() and res.y[props.index('R')] >= tol:

		i = i+1
		res.integrate(res.t+dt)
		sols[0,i] = res.t	# r values
		
		for j in range(len(props)):
		
			sols[j+1,i] = res.y[j]	# p,m + other values
		
	vals = [sols[j,i] for j in range(len(props)+1)] # surface values of R,p,M,Lambda
	
	def Rkm(vals): # R in km
	
		R = vals[0]
	
		return R/1e5
		
	def MMsun(vals): # M in Msun
	
		M = vals[props.index('M')+1]
	
		return M/Msun
		
	def Lambda0(vals): # dimensionless tidal deformability
	
		etaR = vals[props.index('Lambda')+1]
	
		C = G*vals[props.index('M')+1]/(c**2*vals[0])
		fR = 1.-2.*C
	
		F = hyp2f1(3.,5.,6.,2.*C)
		
		def dFdz():

			z = 2.*C

			return (5./(2.*z**6.))*(z*(-60.+z*(150.+z*(-110.+3.*z*(5.+z))))/(z-1.)**3+60.*np.log(1.-z))
	
		RdFdr = -2.*C*dFdz()
		
		k2el = 0.5*(etaR-2.-4.*C/fR)/(RdFdr-F*(etaR+3.-4.*C/fR))
	
		return (2./3.)*(k2el/C**5)
	
	units = dict(R=Rkm,M=MMsun,Lambda=Lambda0)
	
	return [units[prop](vals) for prop in props]

print tov("../eos/pwp_1.csv",6e14,'Lambda,M,R')
