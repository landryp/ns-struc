#!/usr/bin/python

import numpy as np
from optparse import OptionParser
from scipy.interpolate import interp1d

# READ IN INPUT PROPERTY AND TARGET VALUE

def macro(propspath,inprop='M=1.4'):

	var, val = inprop.split('=')

	properties = np.genfromtxt(propspath,names=True,delimiter=',')
	props = list(properties.dtype.names)
	props.remove(var)
	numprops = len(props)
	
	if val == 'max':
	
		val = max(properties(var))
		
	elif val == 'min':
	
		val = min(properties(var))
		
	else:
	
		val = float(val)

# INTERPOLATE MASS RELATIONS FOR EACH EOS
	
	macro = [val]
	i = 0
	for prop in props:
	
		rel = interp1d(properties[var],properties[prop],kind='linear',bounds_error=False,fill_value=0)
		macro.append(float(rel(val)))
		i = i+1

# RETURN PROPERTIES AT TARGET VALUE
	
	return macro

