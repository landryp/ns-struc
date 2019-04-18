#!/usr/bin/python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import ode
from .struceqs import *
from .constants import *

# INTERPOLATE CONTINUOUS FLUID VARIABLES FROM DISCRETE EOS DATA

def tov(eospath,rhoc,props=['R','M'],stp=1e1,pts=1e3,maxr=2e6,tol=1e1):

	pts = int(pts)
	eqs = eqsdict() # associate NS properties with corresponding equation of stellar structure

	if len(eospath) == 3:
	
		mu, P, cs2i = eospath

	else:	
	
		eosdat = np.genfromtxt(eospath,names=True,delimiter=',') # load EoS data
		rhodat = eosdat['baryon_density'] # rest-mass energy density in units of g/cm^3
		pdat = eosdat['pressurec2'] # pressure in units of g/cm^3
		mudat = eosdat['energy_densityc2'] # total energy density in units of g/cm^3

		mup = interp1d(pdat,mudat,kind='linear',bounds_error=False,fill_value=0)
		def mu(p): return mup(p)
		
		prho = interp1d(rhodat,pdat,kind='linear',bounds_error=False,fill_value=0)
		def P(rho): return prho(rho)
		
		cs2pi = interp1d(pdat,np.gradient(mudat,pdat),kind='linear', bounds_error=False, fill_value=0)
		def cs2i(p): return cs2pi(p) # 1/sound speed squared

# PERFORM INTEGRATION OF EQUATIONS OF STELLAR STRUCTURE
		
	def efe(r,y): return [eqs[prop](r,y,mu,cs2i,props) for prop in props]

	pc = float(P(rhoc)) # central pressure from interpolated p(rho) function
	muc = mu(pc) # central energy density from interpolated mu(p) function
	cs2ic = cs2i(pc) # central sound speed from interpolated cs2i(p) function
	startvals = initconds(pc,muc,cs2ic,stp,props) # load BCs at center of star for integration
	y0 = [startvals[prop] for prop in props]
	
	res = ode(efe)
	res.set_initial_value(y0,stp)
	dt = (maxr-stp)/pts # fixed radial step size for data returned by integration
	
	sols = np.zeros((len(props)+1,pts)) # container for solutions
	i=-1
	while res.successful() and res.y[props.index('R')] >= tol and i < pts-1: # stop integration when pressure vanishes (to within tolerance tol)

		i = i+1
		res.integrate(res.t+dt)
		sols[0,i] = res.t	# r values
		
		for j in range(len(props)):
		
			sols[j+1,i] = res.y[j]	# p, m + other values
		
	vals = [sols[j,i] for j in range(len(props)+1)] # surface values of R, p, M, etc.

# CALCULATE NS PROPERTIES AT SURFACE
	
	obs = calcobs(vals,props)
	
	return [obs[prop](vals) for prop in props]

