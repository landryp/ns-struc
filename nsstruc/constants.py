"definition of useful physical constants"
__author__ = "philippe.landry@ligo.org and Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from numpy import pi

#-------------------------------------------------

twopi = 2.*pi
fourpi = 4.*pi

c = 2.99792458e10 # speed of light in cgs
c2 = c**2
G = 6.67408e-8 # Newton's constant in cgs

G_c2 = G/c2

Msun = 1.3271244e26/G # solar mass in cgs
rhonuc = 2.8e14 # nuclear density in cgs
