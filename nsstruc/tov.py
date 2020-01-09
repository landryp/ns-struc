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

