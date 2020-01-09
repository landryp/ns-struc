"a module that hoses logic for TOV integration"
__author__ = 'philippe.landry@ligo.org and Reed Essick (reed.essick@gmail.com)'

#-------------------------------------------------

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import ode

### non-standard libraries
from .struceqs import *
from .constants import *

#-------------------------------------------------

DEFAULT_INITIAL_R = 10 ### cm
DEFAULT_NUM_R = 2000
DEFAULT_MAX_R = 2e6 ### cm
DEFAULT_PRESSUREC2_TOL = 10 ### g/cm^3

DEFAULT_PROPS = ['R', 'M', 'Lambda']
KNOWN_PROPS = ['R', 'M', 'Lambda', 'I', 'Mb']

#-------------------------------------------------
# INTERPOLATE CONTINUOUS FLUID VARIABLES FROM DISCRETE EOS DATA

def tov(eospath,rhoc,props=['R','M','Lambda'],stp=1e1,pts=2e3,maxr=2e6,tol=1e1):

	pts = int(pts)
	eqs = eqsdict() # associate NS properties with corresponding equation of stellar structure

	if len(eospath) == 4: # pass pre-interpolated fluid variables instead of table path
	
		mu, P, cs2i, Rho = eospath

	else:	
	
		eosdat = np.genfromtxt(eospath,names=True,delimiter=',') # load EoS data
		rhodat = eosdat['baryon_density'] # rest-mass energy density in units of g/cm^3
		pdat = eosdat['pressurec2'] # pressure in units of g/cm^3
		mudat = eosdat['energy_densityc2'] # total energy density in units of g/cm^3

		rhop = interp1d(pdat,rhodat,kind='linear',bounds_error=False,fill_value=0)
		def Rho(p): return rhop(p)

		mup = interp1d(pdat,mudat,kind='linear',bounds_error=False,fill_value=0)
		def mu(p): return mup(p)
		
		prho = interp1d(rhodat,pdat,kind='linear',bounds_error=False,fill_value=0)
		def P(rho): return prho(rho)
		
		cs2pi = interp1d(pdat,np.gradient(mudat,pdat),kind='linear', bounds_error=False, fill_value=0)
		def cs2i(p): return cs2pi(p) # 1/sound speed squared

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
	
#	sols = np.zeros((len(props)+1,pts)) # container for solutions
	i=-1
	while res.successful() and res.y[props.index('R')] >= tol and i < pts-1: # stop integration when pressure vanishes (to within tolerance tol)

		i = i+1
		res.integrate(res.t+dt)
#		sols[0,i] = res.t	# r values		# UNCOMMENT TO STORE FULL SOLS
#		sols[1:,i] = res.y	# p, m + other values
		
#	vals = [sols[j,i] for j in range(len(props)+1)] # surface values of R, p, M, etc.
	vals = [res.t] + list(res.y)
	
# CALCULATE NS PROPERTIES AT SURFACE
	
	obs = calcobs(vals,props)
	
	return [obs[prop](vals) for prop in props]

