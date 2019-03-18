#!/usr/bin/python

import numpy as np

def geteos(eosloc):

	eosfile = eosloc.split('/')[-1]
	extension = '.'+eosfile.split('.')[-1]

	if extension == '.dat':
	
		eos = np.genfromtxt(eosloc)
		
	else:
	
		eosdat = np.genfromtxt(eosloc,names=True,delimiter=',')
		p = eosdat['pressurec2']
		mu = eosdat['energy_densityc2']
		rho = eosdat['baryon_density']
	
		eos = np.zeros((len(p),3))
		
		for i in range(len(p)):
		
			eos[i] = [rho[i],p[i],mu[i]]
	
	return eos

