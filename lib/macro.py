#!/usr/bin/python

import numpy as np
from optparse import OptionParser
from scipy.interpolate import interp1d

# INTERPOLATE MASS RELATIONS FOR EACH EOS

def macro(propsloc,prop='M=1.4'):

	var, val = prop.split('=')

	properties = np.genfromtxt(propsloc,names=True,delimiter=',')
	propslist = list(properties.dtype.names)
	
	macro = []
	
	rhocdat = properties['rhoc']
	Mdat = properties['M']
	Rdat = properties['R']
	
	if var == 'M':
	
		if val == 'max':
		
			val = max(Mdat)
			
		elif val == 'min':
		
			val = min(Mdat)
		
		else:
		
			val = float(val)
	
		rhocM = interp1d(Mdat,rhocdat,kind='linear',bounds_error=False,fill_value=0.)	
		RM = interp1d(Mdat,Rdat,kind='linear',bounds_error=False,fill_value=0.)
	
		macro.append(float(rhocM(val)))
		macro.append(val)
		macro.append(float(RM(val)))
	
		if 'Lambda' in propslist:
	
			Lambdadat = properties['Lambda']
		
			LambdaM = interp1d(Mdat,Lambdadat,kind='linear',bounds_error=False,fill_value=0.)
		
			macro.append(float(LambdaM(val)))
			
	if var == 'rhoc':
	
		if val == 'max':
		
			val = max(rhocdat)
			
		elif val == 'min':
		
			val = min(rhocdat)
		
		else:
		
			val = float(val)
	
		Mrhoc = interp1d(rhocdat,Mdat,kind='linear',bounds_error=False,fill_value=0.)	
		Rrhoc = interp1d(rhocdat,Rdat,kind='linear',bounds_error=False,fill_value=0.)
	
		macro.append(val)
		macro.append(float(Mrhoc(val)))
		macro.append(float(Rrhoc(val)))
	
		if 'Lambda' in propslist:
	
			Lambdadat = properties['Lambda']
		
			Lambdarhoc = interp1d(rhocdat,Lambdadat,kind='linear',bounds_error=False,fill_value=0.)
		
			macro.append(float(Lambdarhoc(val)))
			
	if var == 'R':
	
		if val == 'max':
		
			val = max(Rdat)
			
		elif val == 'min':
		
			val = min(Rdat)
		
		else:
		
			val = float(val)
	
		MR = interp1d(Rdat,Mdat,kind='linear',bounds_error=False,fill_value=0.)	
		rhocR = interp1d(Rdat,rhocdat,kind='linear',bounds_error=False,fill_value=0.)
	
		macro.append(float(rhocR(val)))
		macro.append(float(MR(val)))
		macro.append(val)
	
		if 'Lambda' in propslist:
	
			Lambdadat = properties['Lambda']
		
			LambdaR = interp1d(Rdat,Lambdadat,kind='linear',bounds_error=False,fill_value=0.)
		
			macro.append(float(LambdaR(val)))
	
	if var == 'Lambda':
	
		Lambdadat = properties['Lambda']
	
		if val == 'max':
		
			val = max(Lambdadat)
			
		elif val == 'min':
		
			val = min(Lambdadat)
		
		else:
		
			val = float(val)
	
		MLambda = interp1d(Lambdadat,Mdat,kind='linear',bounds_error=False,fill_value=0.)	
		rhocLambda = interp1d(Lambdadat,rhocdat,kind='linear',bounds_error=False,fill_value=0.)
		RLambda = interp1d(Lambdadat,Rdat,kind='linear',bounds_error=False,fill_value=0.)
	
		macro.append(float(rhocLambda(val)))
		macro.append(float(MLambda(val)))
		macro.append(float(RLambda(val)))
		macro.append(val)

# RETURN PROPERTIES AT TARGET VALUE
	
	return macro

