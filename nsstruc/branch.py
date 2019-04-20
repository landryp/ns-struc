#!/usr/bin/python

import numpy as np
from optparse import OptionParser

# IDENTIFY ALL STABLE BRANCHES FOR EACH EOS

def branch(propspath):

	properties = np.genfromtxt(propspath,names=True,delimiter=',')

	rhocdat = properties['rhoc'][::-1] # decreasing rhoc list
	Mdat = properties['M'][::-1] # forward differences skip min mass, not max
	
	dMdrhoc = np.diff(Mdat)/np.diff(rhocdat) # grad of M vs rhoc, positive means stable
	datlen = len(dMdrhoc)
	
	critpts = [0] # list of turning points
	signs = [np.sign(dMdrhoc[0])] # stability of each branch
	for i in range(1,datlen):
	
		if np.sign(dMdrhoc[i]) != np.sign(dMdrhoc[i-1]):

			critpts = np.append(critpts,i)
			signs = np.append(signs,np.sign(dMdrhoc[i]))
	
	critpts = np.append(critpts,datlen) # includes start and end points
	
	return [(datlen-critpts)[::-1],signs[::-1]] # undo flip; stable still positive sign
