#!/usr/bin/python

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import ode, quad, trapz
from .struceqs import *
from .constants import *

# INTERPOLATE CONTINUOUS FLUID VARIABLES FROM DISCRETE EOS DATA

def tov(eospath,rhoc,props=['pR','R','M','Lambda'],idxs={'pR':0,'M':1,'R':2,'Lambda':3},fracstp=1e-8,pts=2e3,tol=1e1):

	eqs = eqsdict() # associate NS properties with corresponding equation of stellar structure

	if len(eospath) == 4: # pass pre-interpolated fluid variables instead of table path
	
		e, P, cs2i, Rho = eospath

	else:	
	
		eosdat = np.genfromtxt(eospath,names=True,delimiter=',') # load EoS data
		rhodat = eosdat['baryon_density'] # rest-mass energy density in units of g/cm^3
		pdat = eosdat['pressurec2'] # pressure in units of g/cm^3
		edat = eosdat['energy_densityc2'] # total energy density in units of g/cm^3

		rhop = interp1d(pdat,rhodat,kind='linear',bounds_error=False,fill_value=0)
		def Rho(p): return rhop(p)

		ep = interp1d(pdat,edat,kind='linear',bounds_error=False,fill_value=0)
		def e(p): return ep(p)
		
		prho = interp1d(rhodat,pdat,kind='linear',bounds_error=False,fill_value=0)
		def P(rho): return prho(rho)
		
		cs2pi = interp1d(pdat,np.gradient(edat,pdat),kind='linear', bounds_error=False, fill_value=0)
		def cs2i(p): return cs2pi(p) # 1/sound speed squared

# PERFORM INTEGRATION OF EQUATIONS OF STELLAR STRUCTURE
		
	def efe(h,y): return [eqs[prop](h,y,e,Rho,cs2i,idxs) for prop in ['pR','M','R','Lambda','I','Mb'] if prop in props]

	pc = float(P(rhoc)) # central pressure from interpolated p(rho) function
	ec = e(pc) # central energy density from interpolated e(p) function
	cs2ic = cs2i(pc) # central sound speed from interpolated cs2i(p) function
	
	plist = np.logspace(np.log(tol),np.log(pc),num=200,base=np.exp(1.))
	dhdp = [1./(e(p)+p) for p in plist]
	hc = trapz(dhdp,plist)
	
	startvals = initconds(hc,pc,ec,rhoc,cs2ic,fracstp) # load BCs at center of star for integration
	y0 = [startvals[prop] for prop in ['pR','M','R','Lambda','I','Mb'] if prop in props]
	
	res = ode(efe)
	res.set_initial_value(y0,(1.-fracstp)*hc)
	dt = (1.-fracstp)*hc/pts # fixed enthalpy step size for data returned by integration
	
	while res.successful() and res.y[0] >= tol and res.t >= dt:

		res.integrate(res.t-dt)

	vals = [res.t] + list(res.y)
	
# CALCULATE NS PROPERTIES AT SURFACE
	
	obs = calcobs(vals,idxs)

	return [obs[prop](vals) for prop in ['pR','M','R','Lambda','I','Mb'] if prop in props]

